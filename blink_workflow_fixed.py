"""
Blink Detection Workflow Orchestrator - FIXED VERSION
======================================================

Fixes:
1. Properly processes IR data and generates ir_blinks.csv
2. Uses 6-second interval threshold instead of 15% rate difference
3. Detects when intervals exceed 6 seconds for considerable time

Usage:
    python blink_workflow_fixed.py --video input.mp4 --ir-data ir_data.csv --output-dir results/
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
import traceback

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
        print("BLINK DETECTION WORKFLOW ORCHESTRATOR (FIXED)")
        print(f"{'='*70}")
        print(f"Video: {video_path}")
        print(f"IR Data: {ir_data_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    
    def step1_process_video(self):
        """
        Step 1: Process video to extract blink timestamps.
        Calls video processing script.
        """
        print("\n[STEP 1] Processing Video for Blink Detection...")
        print("-" * 70)
        
        # Video processing code
        try:
            from blink_detector_accurate_timestamps import AccurateBlinkDetector
            import cv2
            
            detector = AccurateBlinkDetector(ear_threshold=0.18, consecutive_frames=2)
            
            # Process video
            cap = cv2.VideoCapture(self.video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            detector.set_fps(fps)
            
            print(f"  Video FPS: {fps}")
            print(f"  Total frames: {total_frames}")
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detector.process_frame(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)...")
            
            cap.release()
            
            # Export CSV
            video_csv_path = self.output_dir / 'video_blinks.csv'
            detector.export_csv(str(video_csv_path))
            
            # Load as dataframe
            self.video_blinks_df = pd.read_csv(video_csv_path)
            
            print(f"[OK] Video processing complete")
            print(f"   Detected {len(self.video_blinks_df)} blinks")
            print(f"   CSV saved: {video_csv_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error processing video: {e}")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            return False
    
    def step2_process_ir_data(self):
        """
        Step 2: Process IR data to extract blink timestamps.
        """
        print("\n[STEP 2] Processing IR Data for Blink Detection...")
        print("-" * 70)
        
        try:
            # Load IR data
            ir_data = pd.read_csv(self.ir_data_path)
            print(f"  Loaded IR data: {len(ir_data)} rows")
            print(f"  Columns: {list(ir_data.columns)}")
            
            # Show preview
            print(f"\n  First 5 rows:")
            print(ir_data.head().to_string())
            
            # Identify timestamp and blink indicator columns
            timestamp_col = None
            blink_col = None
            
            # Look for timestamp column (case-insensitive)
            for col in ir_data.columns:
                col_lower = col.lower()
                if 'time' in col_lower or 'timestamp' in col_lower or 'date' in col_lower:
                    timestamp_col = col
                    break
            
            # Look for blink indicator column
            for col in ir_data.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['blink', 'ir', 'sensor', 'detect', 'eye']):
                    blink_col = col
                    break
            
            # Handle timestamp column
            if timestamp_col is None:
                print(f"\n  [!]  No timestamp column found. Using row index as time.")
                sampling_rate = 30.0  # Default: 30 Hz
                ir_data['Time_Seconds'] = ir_data.index / sampling_rate
                timestamp_col = 'Time_Seconds'
            else:
                print(f"  Using timestamp column: '{timestamp_col}'")
                try:
                    # Parse timestamps
                    ts_raw = ir_data[timestamp_col].astype(str).str.strip()
                    
                    # Fix malformed timestamps
                    ts_fixed = ts_raw.str.replace(
                        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\d{3,6})$",
                        r"\1.\2",
                        regex=True
                    )
                    
                    parsed = pd.to_datetime(ts_fixed, errors='coerce')
                    
                    # Check for parsing failures
                    failed = parsed.isna().sum()
                    if failed > 0:
                        print(f"  Warning: {failed} timestamps failed to parse")
                    
                    # Convert to seconds from start
                    ir_data['Time_Seconds'] = (parsed - parsed.min()).dt.total_seconds()
                    timestamp_col = 'Time_Seconds'
                    
                except Exception as e:
                    print(f"  Warning: Timestamp parsing failed ({e})")
                    print(f"  Falling back to row index")
                    ir_data['Time_Seconds'] = ir_data.index / 30.0
                    timestamp_col = 'Time_Seconds'
            
            # Handle blink indicator column
            if blink_col is None:
                print(f"\n  [!]  No blink indicator column found automatically.")
                print(f"  Available columns: {list(ir_data.columns)}")
                print(f"\n  Please update the code to specify which column indicates blinks.")
                print(f"  For now, will try to use the second column as blink indicator...")
                
                if len(ir_data.columns) > 1:
                    blink_col = ir_data.columns[1]
                    print(f"  Using column: '{blink_col}'")
                else:
                    print(f"  ERROR: Not enough columns in IR data")
                    return False
            else:
                print(f"  Using blink indicator column: '{blink_col}'")
            
            # Show statistics of blink column
            print(f"\n  Blink column statistics:")
            print(f"    Min: {ir_data[blink_col].min()}")
            print(f"    Max: {ir_data[blink_col].max()}")
            print(f"    Mean: {ir_data[blink_col].mean():.3f}")
            print(f"    Unique values: {ir_data[blink_col].nunique()}")
            
            # Detect blinks
            print(f"\n  Detecting blinks from IR data...")
            blinks = self._detect_blinks_from_ir(ir_data, timestamp_col, blink_col)
            
            if not blinks:
                print(f"\n  [!]  No blinks detected in IR data.")
                return False
            
            # Create DataFrame
            self.ir_blinks_df = pd.DataFrame(blinks)
            
            # Calculate intervals
            self.ir_blinks_df['Interval_From_Previous_Seconds'] = \
                self.ir_blinks_df['Midpoint_Time_Seconds'].astype(float).diff()
            
            # Save to CSV
            ir_csv_path = self.output_dir / 'ir_blinks.csv'
            self.ir_blinks_df.to_csv(ir_csv_path, index=False)
            
            print(f"\n[OK] IR processing complete")
            print(f"   Detected {len(self.ir_blinks_df)} blinks")
            print(f"   CSV saved: {ir_csv_path}")
            print(f"\n  Preview of detected blinks:")
            print(self.ir_blinks_df.head(10).to_string())
            
            return True
                
        except Exception as e:
            print(f"[ERROR] Error processing IR data: {e}")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            return False
    
    def _detect_blinks_from_ir(self, ir_data, timestamp_col, blink_col):
        """
        Detect blinks from IR sensor data.
        """
        blinks = []
        blink_counter = 0
        
        in_blink = False
        blink_start = None
        blink_frames = []
        
        # Determine threshold
        threshold = 0.5
        
        for idx, row in ir_data.iterrows():
            time = row[timestamp_col]
            blink_value = row[blink_col]
            
            # Skip NaN values
            if pd.isna(time) or pd.isna(blink_value):
                continue
            
            # value > threshold means blink
            if blink_value > threshold:
                if not in_blink:
                    # Blink starts
                    in_blink = True
                    blink_start = time
                    blink_frames = [time]
                else:
                    # Blink continues
                    blink_frames.append(time)
            else:
                if in_blink:
                    # Blink ends
                    in_blink = False
                    blink_end = blink_frames[-1]
                    
                    if len(blink_frames) >= 1:
                        blink_counter += 1
                        midpoint = (blink_start + blink_end) / 2.0
                        duration = blink_end - blink_start
                        
                        blinks.append({
                            'Blink_Number': blink_counter,
                            'Start_Time_Seconds': f"{blink_start:.3f}",
                            'End_Time_Seconds': f"{blink_end:.3f}",
                            'Midpoint_Time_Seconds': f"{midpoint:.3f}",
                            'Duration_Seconds': f"{duration:.3f}",
                            'Num_Frames': len(blink_frames)
                        })
        
        # Handle case where blink extends to end of data
        if in_blink and blink_frames:
            blink_counter += 1
            blink_end = blink_frames[-1]
            midpoint = (blink_start + blink_end) / 2.0
            duration = blink_end - blink_start
            
            blinks.append({
                'Blink_Number': blink_counter,
                'Start_Time_Seconds': f"{blink_start:.3f}",
                'End_Time_Seconds': f"{blink_end:.3f}",
                'Midpoint_Time_Seconds': f"{midpoint:.3f}",
                'Duration_Seconds': f"{duration:.3f}",
                'Num_Frames': len(blink_frames)
            })
        
        return blinks
    
    def step3_compare_and_analyze(self):
        """
        Step 3: Compare video and IR blink detection results.
        """
        print("\n[STEP 3] Comparing Video vs IR Blink Detection...")
        print("-" * 70)
        
        if self.video_blinks_df is None or self.ir_blinks_df is None:
            print("[ERROR] Missing blink data. Cannot perform comparison.")
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
        
        print(f"\n[OK] Comparison complete")
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
        
        # Count intervals exceeding 6 seconds
        long_intervals = intervals[intervals > 6.0]
        num_long_intervals = len(long_intervals)
        percent_long_intervals = (num_long_intervals / len(intervals)) * 100 if len(intervals) > 0 else 0
        
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
            'max_duration_seconds': max_duration,
            'intervals_over_6s': num_long_intervals,
            'percent_intervals_over_6s': percent_long_intervals,
            'all_intervals': intervals.tolist()
        }
        
        print(f"    Total blinks: {metrics['total_blinks']}")
        print(f"    Blink rate: {metrics['blink_rate_per_minute']:.2f} blinks/min")
        print(f"    Avg interval: {metrics['avg_interval_seconds']:.3f}s")
        print(f"    Max interval: {metrics['max_interval_seconds']:.3f}s")
        print(f"    Intervals > 6s: {num_long_intervals} ({percent_long_intervals:.1f}%)")
        
        return metrics
    
    def _perform_comparison(self, video_metrics, ir_metrics):
        """
        Compare video vs IR metrics.
        """
        print(f"\n  Performing comparison with 6-second interval threshold...")
        
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
            'video_is_higher': video_interval > ir_interval
        }
        
        # 6-second interval assessment
        video_long_intervals = video_metrics['intervals_over_6s']
        video_percent_long = video_metrics['percent_intervals_over_6s']
        ir_long_intervals = ir_metrics['intervals_over_6s']
        ir_percent_long = ir_metrics['percent_intervals_over_6s']
        
        comparison['long_intervals'] = {
            'video_count': video_long_intervals,
            'video_percent': video_percent_long,
            'ir_count': ir_long_intervals,
            'ir_percent': ir_percent_long,
            'difference_count': video_long_intervals - ir_long_intervals,
            'difference_percent': video_percent_long - ir_percent_long
        }
        
        # Overall assessment using 6-second threshold
        # Blinking is "reduced" if:
        # 1. Video has significantly more intervals (>20%) which are greater than 6s 
        # 2. OR average interval is significantly higher
        
        LONG_INTERVAL_THRESHOLD_PERCENT = 20.0  # 20% of intervals > 6s is concerning
        
        is_reduced = False
        reasons = []
        
        if video_percent_long > LONG_INTERVAL_THRESHOLD_PERCENT:
            is_reduced = True
            reasons.append(f"{video_percent_long:.1f}% of inter-blink intervals exceed 6 seconds")
        
        if video_interval > 5.0:  # Average interval itself is very high
            is_reduced = True
            reasons.append(f"Average inter-blink interval is {video_interval:.2f}s (approaching 6s threshold)")
        
        comparison['assessment'] = {
            'is_blinking_reduced': is_reduced,
            'long_interval_threshold': LONG_INTERVAL_THRESHOLD_PERCENT,
            'six_second_threshold': 6.0,
            'reasons': reasons,
            'conclusion': self._generate_conclusion(is_reduced, reasons, 
                                                   video_rate, ir_rate,
                                                   video_long_intervals, video_percent_long)
        }
        
        print(f"\n  Assessment: {'[!] REDUCED BLINKING' if is_reduced else '[OK] NORMAL BLINKING'}")
        print(f"  Video: {video_long_intervals} intervals > 6s ({video_percent_long:.1f}%)")
        print(f"  IR: {ir_long_intervals} intervals > 6s ({ir_percent_long:.1f}%)")
        
        return comparison
    
    def _generate_conclusion(self, is_reduced, reasons, video_rate, ir_rate, 
                           video_long_intervals, video_percent_long):
        """Generate human-readable conclusion (ASCII-friendly for Windows)."""
        if is_reduced:
            conclusion = f"[!] ALERT: Reduced blinking detected.\n\n"
            conclusion += f"Video shows {video_rate:.1f} blinks/min vs IR baseline of {ir_rate:.1f} blinks/min.\n\n"
            conclusion += f"Critical finding: {video_long_intervals} inter-blink intervals exceeded 6 seconds "
            conclusion += f"({video_percent_long:.1f}% of all intervals).\n\n"
            conclusion += "Reasons:\n"
            for i, reason in enumerate(reasons, 1):
                conclusion += f"  {i}. {reason}\n"
            conclusion += "\nThis may indicate:\n"
            conclusion += "  - Significant digital eye strain\n"
            conclusion += "  - Screen time fatigue\n"
            conclusion += "  - Deep concentration (may be normal)\n"
            conclusion += "  - Dry eye symptoms requiring attention\n"
            conclusion += "\nRecommendation: Consider taking breaks or adjusting screen settings.\n"
        else:
            conclusion = f"[OK] Normal blinking pattern.\n\n"
            conclusion += f"Video shows {video_rate:.1f} blinks/min vs IR baseline of {ir_rate:.1f} blinks/min.\n"
            conclusion += f"Only {video_long_intervals} intervals exceeded 6 seconds ({video_percent_long:.1f}%).\n"
            conclusion += "Blink pattern is within healthy range."
        
        return conclusion
    
    def step4_generate_report(self):
        """
        Step 4: Generate comprehensive report with visualizations.
        """
        print("\n[STEP 4] Generating Report...")
        print("-" * 70)
        
        if not self.comparison_results:
            print("[ERROR] No comparison results available.")
            return False
        
        # Generate visualizations
        self._create_visualizations()
        
        self._create_text_report()
        
        print(f"\n[OK] Report generation complete")
        print(f"   Output directory: {self.output_dir}")
        return True
    
    def _create_visualizations(self):
        """Create simple rectangular pulse timeline visualization."""
        print("  Creating visualizations...")
        
        video_metrics = self.comparison_results['video_metrics']
        ir_metrics = self.comparison_results['ir_metrics']
        assessment = self.comparison_results['comparison']['assessment']
        
        # Create figure with a single subplot (Video timeline only)
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 4), sharex=True)
        
        # Get max time for x-axis
        max_time = max(
            self.video_blinks_df['End_Time_Seconds'].astype(float).max(),
            self.ir_blinks_df['End_Time_Seconds'].astype(float).max()
        )
        
        # Plot 1: Video Blink Pulses
        ax1.set_title('Video Blink Timeline', fontsize=14, fontweight='bold', pad=10)
        ax1.set_ylabel('Blink', fontsize=12, fontweight='bold')
        ax1.set_ylim(-0.1, 1.2)
        ax1.set_xlim(0, max_time)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['', 'BLINK'])
        
        # Draw rectangular pulses for each blink
        for idx, blink in self.video_blinks_df.iterrows():
            start = float(blink['Start_Time_Seconds'])
            end = float(blink['End_Time_Seconds'])
            duration = end - start
            
            # Make pulses visible - minimum width of 0.5% of total time
            min_width = max_time * 0.005
            if duration < min_width:
                # Center the visible pulse around the midpoint
                midpoint = (start + end) / 2
                start = midpoint - min_width / 2
                end = midpoint + min_width / 2
                duration = min_width
            
            # Check if this interval (to next blink) is > 6s
            interval = float(blink.get('Interval_From_Previous_Seconds', 0))
            color = '#FF6B6B' if interval > 6.0 else '#4ECDC4'  # Red if long gap before
            
            # Draw rectangular pulse
            rect = plt.Rectangle((start, 0), duration, 1, 
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax1.add_patch(rect)
        
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axhline(y=0, color='black', linewidth=2)
        
        # Add legend (avoid Unicode)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Normal interval (<6s)'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Long interval (>6s)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # # Plot 2: IR Blink Pulses
        # ax2.set_title('IR Blink Timeline (Baseline)', fontsize=14, fontweight='bold', pad=10)
        # ax2.set_ylabel('Blink', fontsize=12, fontweight='bold')
        # ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        # ax2.set_ylim(-0.1, 1.2)
        # ax2.set_xlim(0, max_time)
        # ax2.set_yticks([0, 1])
        # ax2.set_yticklabels(['', 'BLINK'])
        
        # # Draw rectangular pulses for each blink
        # for idx, blink in self.ir_blinks_df.iterrows():
        #     start = float(blink['Start_Time_Seconds'])
        #     end = float(blink['End_Time_Seconds'])
        #     duration = end - start
            
        #     # Make pulses visible
        #     min_width = max_time * 0.005
        #     if duration < min_width:
        #         midpoint = (start + end) / 2
        #         start = midpoint - min_width / 2
        #         end = midpoint + min_width / 2
        #         duration = min_width
            
        #     # Check if this interval is > 6s
        #     interval = float(blink.get('Interval_From_Previous_Seconds', 0))
        #     color = '#FF6B6B' if interval > 6.0 else '#95E1D3'
            
        #     # Draw rectangular pulse
        #     rect = plt.Rectangle((start, 0), duration, 1, 
        #                         facecolor=color, edgecolor='black', linewidth=1.5)
        #     ax2.add_patch(rect)
        
        # ax2.grid(True, alpha=0.3, axis='x')
        # ax2.axhline(y=0, color='black', linewidth=2)
        
        # # Add assessment text box (avoid Unicode emojis for matplotlib)
        # status_text = "REDUCED BLINKING" if assessment['is_blinking_reduced'] else "NORMAL BLINKING"
        # assessment_text = f"{status_text}\n"
        # assessment_text += f"Video: {video_metrics['intervals_over_6s']} long gaps | "
        # assessment_text += f"IR: {ir_metrics['intervals_over_6s']} long gaps"
        
        # bgcolor = 'lightcoral' if assessment['is_blinking_reduced'] else 'lightgreen'
        # fig.text(0.5, 0.02, assessment_text, ha='center', fontsize=12, fontweight='bold',
        #         bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.7, edgecolor='black', linewidth=2))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        viz_path = self.output_dir / 'blink_timeline.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {viz_path}")
    
    def _create_text_report(self):
        """Create detailed text report."""
        print("  Creating text report...")
        
        report_path = self.output_dir / 'comparison_report.txt'
        
        # Use UTF-8 encoding to handle special characters
        with open(report_path, 'w', encoding='utf-8') as f:
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
            f.write("COMPARISON ANALYSIS (6-Second Threshold)\n")
            f.write("="*70 + "\n\n")
            
            comp = self.comparison_results['comparison']
            
            f.write(f"Blink Rate:\n")
            f.write(f"  Video: {comp['blink_rate']['video']:.2f} blinks/min\n")
            f.write(f"  IR: {comp['blink_rate']['ir']:.2f} blinks/min\n")
            f.write(f"  Difference: {comp['blink_rate']['percent_difference']:+.1f}%\n\n")
            
            f.write(f"Long Intervals (>6 seconds):\n")
            f.write(f"  Video: {comp['long_intervals']['video_count']} intervals ({comp['long_intervals']['video_percent']:.1f}%)\n")
            f.write(f"  IR: {comp['long_intervals']['ir_count']} intervals ({comp['long_intervals']['ir_percent']:.1f}%)\n")
            f.write(f"  Difference: {comp['long_intervals']['difference_count']:+d} intervals\n\n")
            
            f.write("="*70 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*70 + "\n\n")
            # Replace Unicode emojis with ASCII
            conclusion = comp['assessment']['conclusion']
            conclusion = conclusion.replace('[!]', '[!]').replace('[OK]', '[OK]')
            f.write(conclusion)
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
        file.write(f"Intervals > 6s: {metrics['intervals_over_6s']} ({metrics['percent_intervals_over_6s']:.1f}%)\n")
        if metrics['avg_duration_seconds']:
            file.write(f"Average Blink Duration: {metrics['avg_duration_seconds']:.3f}s\n")
    

    def run_complete_workflow(self):
        """Execute the complete workflow."""
        print("\n[START] Starting Complete Workflow...\n")
        
        if not self.step1_process_video():
            print("\n[ERROR] Workflow failed at Step 1")
            return False
        
        if not self.step2_process_ir_data():
            print("\n[ERROR] Workflow failed at Step 2")
            return False
        
        if not self.step3_compare_and_analyze():
            print("\n[ERROR] Workflow failed at Step 3")
            return False
        
        if not self.step4_generate_report():
            print("\n[ERROR] Workflow failed at Step 4")
            return False
        
        print("\n" + "="*70)
        print("[SUCCESS] WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n[FILES] All results saved to: {self.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - video_blinks.csv")
        print(f"  - ir_blinks.csv")
        print(f"  - blink_timeline.png")
        print(f"  - comparison_report.txt")
        print(f"  - comparison_report.json")
        
        assessment = self.comparison_results['comparison']['assessment']
        long_int = self.comparison_results['comparison']['long_intervals']
        
        print(f"\n{'='*70}")
        print(f"KEY FINDING:")
        print(f"{'='*70}")
        if assessment['is_blinking_reduced']:
            print(f"[!]  REDUCED BLINKING DETECTED")
            print(f"   Video: {long_int['video_count']} intervals exceeded 6 seconds ({long_int['video_percent']:.1f}%)")
            for reason in assessment['reasons']:
                print(f"   • {reason}")
        else:
            print(f"[OK] Normal blinking pattern detected")
            print(f"   Video: Only {long_int['video_count']} intervals exceeded 6 seconds ({long_int['video_percent']:.1f}%)")
        print(f"{'='*70}\n")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Blink Detection Workflow Orchestrator (Fixed Version)'
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--ir-data', required=True, help='Path to IR data CSV')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Error: Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.ir_data):
        print(f"[ERROR] Error: IR data file not found: {args.ir_data}")
        return 1
    
    orchestrator = BlinkWorkflowOrchestrator(
        video_path=args.video,
        ir_data_path=args.ir_data,
        output_dir=args.output_dir
    )
    
    success = orchestrator.run_complete_workflow()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
