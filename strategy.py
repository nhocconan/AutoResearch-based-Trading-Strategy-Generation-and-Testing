# This strategy uses 4-hour timeframe with a focus on the 123 reversal pattern, combined with 1-day trend filter and volume confirmation.
# The 123 pattern is a swing-based reversal setup: point 1 (swing extreme), point 2 (pullback), point 3 (failure to exceed point 1).
# It works in both bull and bear markets by trading with the higher timeframe trend.
# Volume confirms momentum behind the breakout. Target: 25-40 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_123Reversal_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate swing highs/lows for 123 pattern (using 5-bar window)
    def find_swing_points(high, low, window=2):
        """Find swing highs and lows"""
        n = len(high)
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)
        
        for i in range(window, n - window):
            # Swing high: highest high in window
            if high[i] == np.max(high[i-window:i+window+1]):
                swing_high[i] = high[i]
            # Swing low: lowest low in window
            if low[i] == np.min(low[i-window:i+window+1]):
                swing_low[i] = low[i]
        return swing_high, swing_low
    
    swing_high, swing_low = find_swing_points(high, low, 2)
    
    # Track last swing points
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_swing_high_idx = np.full(n, -1)
    last_swing_low_idx = np.full(n, -1)
    
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_swing_high[i] = swing_high[i]
            last_swing_high_idx[i] = i
        elif i > 0:
            last_swing_high[i] = last_swing_high[i-1]
            last_swing_high_idx[i] = last_swing_high_idx[i-1]
            
        if not np.isnan(swing_low[i]):
            last_swing_low[i] = swing_low[i]
            last_swing_low_idx[i] = i
        elif i > 0:
            last_swing_low[i] = last_swing_low[i-1]
            last_swing_low_idx[i] = last_swing_low_idx[i-1]
    
    # Align daily trend to 4h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get today's daily data for trend
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_today = ema20_1d[idx_1d]
        vol_avg_today = vol_avg_20[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirmed = vol_current > 1.3 * vol_avg_today
        
        price = close[i]
        
        if position == 0:
            # Look for 123 reversal patterns
            
            # Check for long 123 pattern
            # Need: swing low (point1), pullback to point2, failure at point3, break above point2
            if last_swing_low_idx[i] >= 20:  # Ensure we have history
                point1_idx = last_swing_low_idx[i]
                point1 = last_swing_low[i]
                
                # Find point2: pullback high after point1
                point2_idx = -1
                point2 = -1
                for j in range(point1_idx + 1, min(i, point1_idx + 20)):
                    if high[j] > point1 and (point2 == -1 or high[j] < point2):
                        point2 = high[j]
                        point2_idx = j
                
                if point2_idx != -1 and point2 > point1:
                    # Find point3: failure low after point2 (should not exceed point1)
                    point3_idx = -1
                    point3 = 1e10
                    for j in range(point2_idx + 1, i):
                        if low[j] < point2 and low[j] > point1 and low[j] < point3:
                            point3 = low[j]
                            point3_idx = j
                    
                    if point3_idx != -1 and point3 < point2:
                        # Check for break above point2 (entry)
                        if price > point2 and vol_confirmed and price > ema20_today:
                            signals[i] = 0.25
                            position = 1
                            continue
            
            # Check for short 123 pattern
            # Need: swing high (point1), pullback to point2, failure at point3, break below point2
            if last_swing_high_idx[i] >= 20:
                point1_idx = last_swing_high_idx[i]
                point1 = last_swing_high[i]
                
                # Find point2: pullback low after point1
                point2_idx = -1
                point2 = 1e10
                for j in range(point1_idx + 1, min(i, point1_idx + 20)):
                    if low[j] < point1 and (point2 == 1e10 or low[j] > point2):
                        point2 = low[j]
                        point2_idx = j
                
                if point2_idx != -1 and point2 < point1:
                    # Find point3: failure high after point2 (should not go below point1)
                    point3_idx = -1
                    point3 = -1
                    for j in range(point2_idx + 1, i):
                        if high[j] > point2 and high[j] < point1 and high[j] > point3:
                            point3 = high[j]
                            point3_idx = j
                    
                    if point3_idx != -1 and point3 > point2:
                        # Check for break below point2 (entry)
                        if price < point2 and vol_confirmed and price < ema20_today:
                            signals[i] = -0.25
                            position = -1
                            continue
        
        elif position == 1:
            # Exit long: break below point2 of the pattern or trend change
            exit_signal = False
            if price < ema20_today:  # Trend change
                exit_signal = True
            elif not vol_confirmed:  # Volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above point2 of the pattern or trend change
            exit_signal = False
            if price > ema20_today:  # Trend change
                exit_signal = True
            elif not vol_confirmed:  # Volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals