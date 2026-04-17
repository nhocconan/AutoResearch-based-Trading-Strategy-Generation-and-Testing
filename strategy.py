#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 123-reversal pattern with 1d volume confirmation.
# The 123-reversal (also called 1-2-3 pattern) identifies trend exhaustion:
# Point 1: swing extreme, Point 2: pullback, Point 3: failed retest of Point 1.
# Entry: break of Point 2 with volume confirmation on 1d timeframe.
# Exit: opposite 123 pattern forms or volatility contraction.
# Works in trending and ranging markets by capturing exhaustion at swing points.

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for swing detection
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on daily data
    volume_1d_series = pd.Series(volume_1d)
    vol_avg20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # Find swing points (simplified fractal detection)
    def find_swing_points(high, low, left=2, right=2):
        """Find swing highs and lows"""
        n = len(high)
        swing_high = np.zeros(n, dtype=bool)
        swing_low = np.zeros(n, dtype=bool)
        
        for i in range(left, n - right):
            # Swing high: highest in window
            if high[i] == np.max(high[i-left:i+right+1]):
                swing_high[i] = True
            # Swing low: lowest in window
            if low[i] == np.min(low[i-left:i+right+1]):
                swing_low[i] = True
        return swing_high, swing_low
    
    swing_high, swing_low = find_swing_points(high, low, 2, 2)
    
    # Track most recent swing points for 123 pattern
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    last_swing_high_val = 0
    last_swing_low_val = 0
    
    # Track Point 2 (pullback) for each swing
    point2_high_idx = -1  # For bearish pattern (after swing high)
    point2_low_idx = -1   # For bullish pattern (after swing low)
    point2_high_val = 0
    point2_low_val = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if np.isnan(vol_avg20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update swing points
        if swing_high[i]:
            last_swing_high_idx = i
            last_swing_high_val = high[i]
            # Reset bearish pattern tracking
            point2_high_idx = -1
            point2_high_val = 0
        
        if swing_low[i]:
            last_swing_low_idx = i
            last_swing_low_val = low[i]
            # Reset bullish pattern tracking
            point2_low_idx = -1
            point2_low_val = 0
        
        # Track pullbacks (Point 2) after swing points
        if last_swing_high_idx != -1 and i > last_swing_high_idx:
            # Looking for pullback after swing high (for bearish 123)
            if low[i] < low[i-1] and low[i] < low[i+1] if i+1 < n else True:
                # Simple pullback detection: local low
                if point2_high_idx == -1 or low[i] < low[point2_high_idx]:
                    point2_high_idx = i
                    point2_high_val = low[i]
        
        if last_swing_low_idx != -1 and i > last_swing_low_idx:
            # Looking for pullback after swing low (for bullish 123)
            if high[i] > high[i-1] and high[i] > high[i+1] if i+1 < n else True:
                # Simple pullback detection: local high
                if point2_low_idx == -1 or high[i] > high[point2_low_idx]:
                    point2_low_idx = i
                    point2_low_val = high[i]
        
        vol_1d_current = volume_1d[i // (24*60//4)] if hasattr(volume_1d, '__getitem__') else 0
        # Actually get the aligned volume for current bar
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Check for bullish 123 pattern completion (break of Point 2 after swing low)
        bullish_setup = (
            last_swing_low_idx != -1 and 
            point2_low_idx != -1 and 
            i > point2_low_idx and
            close[i] > point2_low_val and  # Break above Point 2
            close[i-1] <= point2_low_val   # Was at or below Point 2
        )
        
        # Check for bearish 123 pattern completion (break of Point 2 after swing high)
        bearish_setup = (
            last_swing_high_idx != -1 and 
            point2_high_idx != -1 and 
            i > point2_high_idx and
            close[i] < point2_high_val and  # Break below Point 2
            close[i-1] >= point2_high_val   # Was at or above Point 2
        )
        
        if position == 0:
            # Bullish entry: break of Point 2 after swing low + volume
            if bullish_setup and vol_filter:
                signals[i] = 0.25
                position = 1
            # Bearish entry: break of Point 2 after swing high + volume
            elif bearish_setup and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish 123 forms or volatility drop
            if bearish_setup and vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish 123 forms or volatility drop
            if bullish_setup and vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_123Reversal_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0