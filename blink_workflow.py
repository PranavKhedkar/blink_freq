"""
Blink Detection Workflow Orchestrator
======================================

Orchestrates the complete pipeline:
1. Process video → Extract blink timestamps
2. Process IR data → Extract blink timestamps
3. Compare both sources → Determine if blinking is reduced
4. Generate comprehensive report

Usage:
    python blink_workflow.py --video input.mp4 --ir-data ir_data.csv --output-dir results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json

# Workflow orchestrator
class BlinkWorkflowOrchestrator:
    """
    Orchestrates the complete blink detection and comparison workflow.
    """
    
    def __init__(self, video_path, ir_data_path, output_dir='results'):
        self.video_path = video_path
        self.ir_data_path = ir_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.video_blinks_df = None
        self.ir_blinks_df = None
        self.comparison_results = {}
        
        print(f"\n{'='*70}")
        print("BLINK DETECTION WORKFLOW ORCHESTRATOR")
        print(f"{'='*70}")
        print(f"Video: {video_path}")
        print(f"IR Data: {ir_data_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    
    def step1_process_video(self):
        """
        Step 1: Process video to extract blink timestamps.
        Calls your video processing script.
        """
        print("\n[STEP 1] Processing Video for Blink Detection...")
        print("-" * 70)
        
        # Import your video processing code
        try:
            # This would call your blink detection notebook/script
            # For now, simulating the call
            from blink_detector_accurate_timestamps import AccurateBlinkDetector
            import cv2
            
            detector = AccurateBlinkDetector(ear_threshold=0.18, consecutive_frames=2)
            
            # Process video
            cap = cv2.VideoCapture(self.video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            detector.set_fps(fps)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detector.process_frame(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames...")
            
            cap.release()
            
            # Export CSV
            video_csv_path = self.output_dir / 'video_blinks.csv'
            detector.export_csv(str(video_csv_path))
            
            # Load as dataframe
            self.video_blinks_df = pd.read_csv(video_csv_path)
            
            print(f"✅ Video processing complete")
            print(f"   Detected {len(self.video_blinks_df)} blinks")
            print(f"   CSV saved: {video_csv_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return False
    
    def step2_process_ir_data(self):
        """
        Step 2: Process IR data to extract blink timestamps.
        Calls your IR processing script.
        """
        print("\n[STEP 2] Processing IR Data for Blink Detection...")
        print("-" * 70)
        
        try:
            # Load IR data
            ir_data = pd.read_csv(self.ir_data_path)
            print(f"  Loaded IR data: {len(ir_data)} rows")

            # Ensure Timestamp is string and strip whitespace
            if 'Timestamp' not in ir_data.columns:
                raise KeyError("IR data must contain a 'Timestamp' column")

            ts_raw = ir_data['Timestamp'].astype(str).str.strip()

            # Fix common malformed pattern where microseconds are concatenated
            # e.g. '2025-11-10T19:55:1009656' -> '2025-11-10T19:55:10.09656'
            ts_fixed = ts_raw.str.replace(
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\d{3,6})$",
                r"\1.\2",
                regex=True
            )

            # Try to parse timestamps robustly. Use ISO8601/mixed inference where available,
            # fall back to infer_datetime_format and coerce errors to NaT so we can inspect them.
            try:
                print("  Attempting to parse timestamps with ISO8601 format...")
                parsed = pd.to_datetime(ts_fixed, format='ISO8601', errors='coerce')
            except Exception:
                print("  ISO8601 parsing failed, falling back to infer_datetime_format...")
                parsed = pd.to_datetime(ts_fixed, errors='coerce', infer_datetime_format=True)

            # Report any parsing failures to help debugging
            bad_mask = parsed.isna() & ts_fixed.notna()
            if bad_mask.any():
                n_bad = bad_mask.sum()
                print(f"  Warning: {n_bad} timestamps failed to parse. Examples:")
                print(ir_data.loc[bad_mask, 'Timestamp'].head(10).to_list())

            # Assign parsed timestamps back and preserve previous formatting behaviour
            ir_data['Timestamp'] = parsed
            ir_data['Timestamp'] = ir_data['Timestamp'].dt.strftime("%H:%M:%S.%f").str[:-4]
            
            ir_csv_path = self.output_dir / 'ir_blinks.csv'
            
            # For demonstration, assuming IR script outputs similar format
            # Load the IR blinks CSV
            if ir_csv_path.exists():
                self.ir_blinks_df = pd.read_csv(ir_csv_path)
                print(f"✅ IR processing complete")
                print(f"   Detected {len(self.ir_blinks_df)} blinks")
                print(f"   CSV saved: {ir_csv_path}")
                return True
            else:
                print(f"⚠️  IR blinks CSV not found. Please ensure IR processing creates: {ir_csv_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing IR data: {e}")
            return False
    
    def step3_compare_and_analyze(self):
        """
        Step 3: Compare video and IR blink detection results.
        Determine if user is blinking less than normal.
        """
        print("\n[STEP 3] Comparing Video vs IR Blink Detection...")
        print("-" * 70)
        
        if self.video_blinks_df is None or self.ir_blinks_df is None:
            print("❌ Missing blink data. Cannot perform comparison.")
            return False
        
        # Calculate metrics for video blinks
        video_metrics = self._calculate_metrics(self.video_blinks_df, "Video")
        
        # Calculate metrics for IR blinks
        ir_metrics = self._calculate_metrics(self.ir_blinks_df, "IR")
        
        # Compare the two
        comparison = self._perform_comparison(video_metrics, ir_metrics)
        
        # Store results
        self.comparison_results = {
            'video_metrics': video_metrics,
            'ir_metrics': ir_metrics,
            'comparison': comparison
        }
        
        print(f"\n✅ Comparison complete")
        return True
    
    def _calculate_metrics(self, blinks_df, source_name):
        """Calculate blink metrics from dataframe."""
        print(f"\n  Analyzing {source_name} data...")
        
        # Ensure intervals are calculated
        if 'Interval_From_Previous_Seconds' not in blinks_df.columns:
            if 'Midpoint_Time_Seconds' in blinks_df.columns:
                blinks_df['Interval_From_Previous_Seconds'] = \
                    blinks_df['Midpoint_Time_Seconds'].astype(float).diff()
            else:
                # Use whatever timestamp column exists
                time_col = [col for col in blinks_df.columns if 'Time' in col and 'Seconds' in col][0]
                blinks_df['Interval_From_Previous_Seconds'] = \
                    blinks_df[time_col].astype(float).diff()
        
        intervals = blinks_df['Interval_From_Previous_Seconds'].dropna()
        
        # Get duration info if available
        if 'Duration_Seconds' in blinks_df.columns:
            durations = blinks_df['Duration_Seconds'].astype(float)
            avg_duration = durations.mean()
            min_duration = durations.min()
            max_duration = durations.max()
        else:
            avg_duration = min_duration = max_duration = None
        
        # Calculate total time
        if 'Midpoint_Time_Seconds' in blinks_df.columns:
            total_time = blinks_df['Midpoint_Time_Seconds'].astype(float).max()
        else:
            time_col = [col for col in blinks_df.columns if 'Time' in col and 'Seconds' in col][0]
            total_time = blinks_df[time_col].astype(float).max()
        
        metrics = {
            'total_blinks': len(blinks_df),
            'total_time_seconds': total_time,
            'blink_rate_per_minute': (len(blinks_df) / total_time) * 60,
            'avg_interval_seconds': intervals.mean(),
            'min_interval_seconds': intervals.min(),
            'max_interval_seconds': intervals.max(),
            'std_interval_seconds': intervals.std(),
            'avg_duration_seconds': avg_duration,
            'min_duration_seconds': min_duration,
            'max_duration_seconds': max_duration
        }
        
        print(f"    Total blinks: {metrics['total_blinks']}")
        print(f"    Blink rate: {metrics['blink_rate_per_minute']:.2f} blinks/min")
        print(f"    Avg interval: {metrics['avg_interval_seconds']:.3f}s")
        
        return metrics
    
    def _perform_comparison(self, video_metrics, ir_metrics):
        """Compare video vs IR metrics and determine if blinking is reduced."""
        print(f"\n  Performing comparison...")
        
        comparison = {}
        
        # Compare blink rates
        video_rate = video_metrics['blink_rate_per_minute']
        ir_rate = ir_metrics['blink_rate_per_minute']
        rate_difference = video_rate - ir_rate
        rate_percent_diff = (rate_difference / ir_rate) * 100 if ir_rate > 0 else 0
        
        comparison['blink_rate'] = {
            'video': video_rate,
            'ir': ir_rate,
            'difference': rate_difference,
            'percent_difference': rate_percent_diff,
            'video_is_lower': video_rate < ir_rate
        }
        
        # Compare intervals
        video_interval = video_metrics['avg_interval_seconds']
        ir_interval = ir_metrics['avg_interval_seconds']
        interval_difference = video_interval - ir_interval
        interval_percent_diff = (interval_difference / ir_interval) * 100 if ir_interval > 0 else 0
        
        comparison['interval'] = {
            'video': video_interval,
            'ir': ir_interval,
            'difference': interval_difference,
            'percent_difference': interval_percent_diff,
            'video_is_higher': video_interval > ir_interval  # Higher interval = less frequent
        }
        
        # Overall assessment
        # Blinking is considered "reduced" if:
        # 1. Blink rate is significantly lower (>15% reduction)
        # 2. OR intervals are significantly longer (>15% increase)
        
        THRESHOLD_PERCENT = 15.0
        
        is_reduced = False
        reasons = []
        
        if comparison['blink_rate']['video_is_lower'] and \
           abs(rate_percent_diff) > THRESHOLD_PERCENT:
            is_reduced = True
            reasons.append(f"Blink rate is {abs(rate_percent_diff):.1f}% lower in video")
        
        if comparison['interval']['video_is_higher'] and \
           abs(interval_percent_diff) > THRESHOLD_PERCENT:
            is_reduced = True
            reasons.append(f"Inter-blink interval is {abs(interval_percent_diff):.1f}% longer in video")
        
        comparison['assessment'] = {
            'is_blinking_reduced': is_reduced,
            'threshold_percent': THRESHOLD_PERCENT,
            'reasons': reasons,
            'conclusion': self._generate_conclusion(is_reduced, reasons, 
                                                   video_rate, ir_rate)
        }
        
        print(f"\n  Assessment: {'⚠️ REDUCED BLINKING' if is_reduced else '✅ NORMAL BLINKING'}")
        print(f"  Video rate: {video_rate:.2f} blinks/min")
        print(f"  IR rate: {ir_rate:.2f} blinks/min")
        print(f"  Difference: {rate_percent_diff:+.1f}%")
        
        return comparison
    
    def _generate_conclusion(self, is_reduced, reasons, video_rate, ir_rate):
        """Generate human-readable conclusion."""
        if is_reduced:
            conclusion = f"⚠️ ALERT: Reduced blinking detected.\n\n"
            conclusion += f"Video shows {video_rate:.1f} blinks/min vs IR baseline of {ir_rate:.1f} blinks/min.\n\n"
            conclusion += "Reasons:\n"
            for i, reason in enumerate(reasons, 1):
                conclusion += f"  {i}. {reason}\n"
            conclusion += "\nThis may indicate:\n"
            conclusion += "  - Digital eye strain\n"
            conclusion += "  - Screen time fatigue\n"
            conclusion += "  - Concentration/focus (normal)\n"
            conclusion += "  - Dry eye symptoms\n"
        else:
            conclusion = f"✅ Normal blinking pattern.\n\n"
            conclusion += f"Video shows {video_rate:.1f} blinks/min vs IR baseline of {ir_rate:.1f} blinks/min.\n"
            conclusion += "Blink rate is within normal range."
        
        return conclusion
    
    def step4_generate_report(self):
        """
        Step 4: Generate comprehensive report with visualizations.
        """
        print("\n[STEP 4] Generating Report...")
        print("-" * 70)
        
        if not self.comparison_results:
            print("❌ No comparison results available.")
            return False
        
        # Generate visualizations
        self._create_visualizations()
        
        # Generate text report
        self._create_text_report()
        
        # Generate JSON report
        self._create_json_report()
        
        print(f"\n✅ Report generation complete")
        print(f"   Output directory: {self.output_dir}")
        return True
    
    def _create_visualizations(self):
        """Create comparison visualizations."""
        print("  Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        video_metrics = self.comparison_results['video_metrics']
        ir_metrics = self.comparison_results['ir_metrics']
        
        # Plot 1: Blink Rate Comparison
        ax1 = axes[0, 0]
        categories = ['Video', 'IR']
        rates = [video_metrics['blink_rate_per_minute'], 
                ir_metrics['blink_rate_per_minute']]
        colors = ['#FF6B6B' if rates[0] < rates[1] else '#4ECDC4', '#95E1D3']
        ax1.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Blinks per Minute', fontsize=12, fontweight='bold')
        ax1.set_title('Blink Rate Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (cat, rate) in enumerate(zip(categories, rates)):
            ax1.text(i, rate + 0.5, f'{rate:.1f}', ha='center', fontweight='bold')
        
        # Plot 2: Average Interval Comparison
        ax2 = axes[0, 1]
        intervals = [video_metrics['avg_interval_seconds'], 
                    ir_metrics['avg_interval_seconds']]
        colors = ['#FF6B6B' if intervals[0] > intervals[1] else '#4ECDC4', '#95E1D3']
        ax2.bar(categories, intervals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Average Interval (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Inter-Blink Interval Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (cat, interval) in enumerate(zip(categories, intervals)):
            ax2.text(i, interval + 0.05, f'{interval:.2f}s', ha='center', fontweight='bold')
        
        # Plot 3: Blink Timeline (Video)
        ax3 = axes[1, 0]
        if 'Midpoint_Time_Seconds' in self.video_blinks_df.columns:
            video_times = self.video_blinks_df['Midpoint_Time_Seconds'].astype(float)
            ax3.scatter(video_times, [1]*len(video_times), color='#FF6B6B', 
                       s=100, marker='|', linewidths=3, label='Video Blinks')
            ax3.set_xlabel('Time (seconds)', fontsize=12)
            ax3.set_title('Video Blink Timeline', fontsize=14, fontweight='bold')
            ax3.set_yticks([])
            ax3.grid(True, alpha=0.3, axis='x')
            ax3.legend()
        
        # Plot 4: Assessment Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        assessment = self.comparison_results['comparison']['assessment']
        
        # Create text summary
        summary_text = f"ASSESSMENT SUMMARY\n{'='*40}\n\n"
        summary_text += f"Status: {'⚠️ REDUCED BLINKING' if assessment['is_blinking_reduced'] else '✅ NORMAL'}\n\n"
        summary_text += f"Video Blink Rate: {video_metrics['blink_rate_per_minute']:.2f} blinks/min\n"
        summary_text += f"IR Blink Rate: {ir_metrics['blink_rate_per_minute']:.2f} blinks/min\n"
        summary_text += f"Difference: {self.comparison_results['comparison']['blink_rate']['percent_difference']:+.1f}%\n\n"
        summary_text += f"Video Avg Interval: {video_metrics['avg_interval_seconds']:.2f}s\n"
        summary_text += f"IR Avg Interval: {ir_metrics['avg_interval_seconds']:.2f}s\n"
        summary_text += f"Difference: {self.comparison_results['comparison']['interval']['percent_difference']:+.1f}%\n\n"
        
        if assessment['reasons']:
            summary_text += f"Reasons:\n"
            for reason in assessment['reasons']:
                summary_text += f"  • {reason}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        viz_path = self.output_dir / 'comparison_report.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {viz_path}")
    
    def _create_text_report(self):
        """Create detailed text report."""
        print("  Creating text report...")
        
        report_path = self.output_dir / 'comparison_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BLINK DETECTION COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"IR Data: {self.ir_data_path}\n\n")
            
            f.write("="*70 + "\n")
            f.write("VIDEO BLINK METRICS\n")
            f.write("="*70 + "\n")
            self._write_metrics(f, self.comparison_results['video_metrics'])
            
            f.write("\n" + "="*70 + "\n")
            f.write("IR BLINK METRICS\n")
            f.write("="*70 + "\n")
            self._write_metrics(f, self.comparison_results['ir_metrics'])
            
            f.write("\n" + "="*70 + "\n")
            f.write("COMPARISON ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            comp = self.comparison_results['comparison']
            
            f.write(f"Blink Rate:\n")
            f.write(f"  Video: {comp['blink_rate']['video']:.2f} blinks/min\n")
            f.write(f"  IR: {comp['blink_rate']['ir']:.2f} blinks/min\n")
            f.write(f"  Difference: {comp['blink_rate']['percent_difference']:+.1f}%\n\n")
            
            f.write(f"Inter-Blink Interval:\n")
            f.write(f"  Video: {comp['interval']['video']:.3f}s\n")
            f.write(f"  IR: {comp['interval']['ir']:.3f}s\n")
            f.write(f"  Difference: {comp['interval']['percent_difference']:+.1f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*70 + "\n\n")
            f.write(comp['assessment']['conclusion'])
            f.write("\n\n")
        
        print(f"    Saved: {report_path}")
    
    def _write_metrics(self, file, metrics):
        """Write metrics to file."""
        file.write(f"Total Blinks: {metrics['total_blinks']}\n")
        file.write(f"Total Time: {metrics['total_time_seconds']:.2f}s\n")
        file.write(f"Blink Rate: {metrics['blink_rate_per_minute']:.2f} blinks/min\n")
        file.write(f"Average Interval: {metrics['avg_interval_seconds']:.3f}s\n")
        file.write(f"Min Interval: {metrics['min_interval_seconds']:.3f}s\n")
        file.write(f"Max Interval: {metrics['max_interval_seconds']:.3f}s\n")
        file.write(f"Std Dev Interval: {metrics['std_interval_seconds']:.3f}s\n")
        if metrics['avg_duration_seconds']:
            file.write(f"Average Blink Duration: {metrics['avg_duration_seconds']:.3f}s\n")
    
    def _create_json_report(self):
        """Create machine-readable JSON report."""
        print("  Creating JSON report...")
        
        json_path = self.output_dir / 'comparison_report.json'
        
        # Convert to JSON-serializable format
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'video_path': str(self.video_path),
                'ir_data_path': str(self.ir_data_path)
            },
            'video_metrics': self.comparison_results['video_metrics'],
            'ir_metrics': self.comparison_results['ir_metrics'],
            'comparison': self.comparison_results['comparison']
        }
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"    Saved: {json_path}")
    
    def run_complete_workflow(self):
        """
        Execute the complete workflow.
        """
        print("\n🚀 Starting Complete Workflow...\n")
        
        # Step 1: Process video
        if not self.step1_process_video():
            print("\n❌ Workflow failed at Step 1")
            return False
        
        # Step 2: Process IR data
        if not self.step2_process_ir_data():
            print("\n❌ Workflow failed at Step 2")
            return False
        
        # Step 3: Compare and analyze
        if not self.step3_compare_and_analyze():
            print("\n❌ Workflow failed at Step 3")
            return False
        
        # Step 4: Generate report
        if not self.step4_generate_report():
            print("\n❌ Workflow failed at Step 4")
            return False
        
        print("\n" + "="*70)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 All results saved to: {self.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - video_blinks.csv")
        print(f"  - ir_blinks.csv")
        print(f"  - comparison_report.png")
        print(f"  - comparison_report.txt")
        print(f"  - comparison_report.json")
        
        # Print key finding
        assessment = self.comparison_results['comparison']['assessment']
        print(f"\n{'='*70}")
        print(f"KEY FINDING:")
        print(f"{'='*70}")
        if assessment['is_blinking_reduced']:
            print(f"⚠️  REDUCED BLINKING DETECTED")
            for reason in assessment['reasons']:
                print(f"   • {reason}")
        else:
            print(f"✅ Normal blinking pattern detected")
        print(f"{'='*70}\n")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Blink Detection Workflow Orchestrator'
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--ir-data', required=True, help='Path to IR data CSV')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.ir_data):
        print(f"❌ Error: IR data file not found: {args.ir_data}")
        return 1
    
    # Run workflow
    orchestrator = BlinkWorkflowOrchestrator(
        video_path=args.video,
        ir_data_path=args.ir_data,
        output_dir=args.output_dir
    )
    
    success = orchestrator.run_complete_workflow()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
