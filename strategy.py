#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h 123 Reversal pattern with 1d EMA34 trend filter and volume confirmation
    # 123 Reversal identifies trend exhaustion: new high/low followed by failed continuation
    # Entry: 123 pattern completion + volume spike + 1d EMA34 trend alignment
    # Works in both bull/bear by catching reversals at key swing points
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Swing point detection (lookback 5 periods)
    def find_swing_high(arr, lookback=5):
        highs = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.max(arr[i-lookback:i+lookback+1]):
                highs[i] = arr[i]
        return highs
    
    def find_swing_low(arr, lookback=5):
        lows = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.min(arr[i-lookback:i+lookback+1]):
                lows[i] = arr[i]
        return lows
    
    swing_highs = find_swing_high(high, 5)
    swing_lows = find_swing_low(low, 5)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            i < 10):  # Need lookback for swing detection
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for 123 pattern completion
            # Bullish 123: swing low, higher low, break above swing high
            # Bearish 123: swing high, lower high, break below swing low
            
            # Check for bullish setup
            if i >= 2:
                # Find recent swing low
                recent_swing_low_idx = -1
                for j in range(i-1, max(0, i-20), -1):
                    if not np.isnan(swing_lows[j]):
                        recent_swing_low_idx = j
                        break
                
                if recent_swing_low_idx != -1 and i - recent_swing_low_idx >= 3:
                    # Check for higher low after swing low
                    higher_low_found = False
                    for j in range(recent_swing_low_idx + 1, i):
                        if low[j] > low[recent_swing_low_idx]:
                            higher_low_found = True
                            break
                    
                    if higher_low_found:
                        # Check for break above recent swing high
                        recent_swing_high_idx = -1
                        for j in range(recent_swing_low_idx, i):
                            if not np.isnan(swing_highs[j]):
                                recent_swing_high_idx = j
                                break
                        
                        if (recent_swing_high_idx != -1 and 
                            close[i] > high[recent_swing_high_idx] and
                            vol_spike[i] and 
                            close[i] > ema34_1d_aligned[i]):
                            signals[i] = 0.25
                            position = 1
                            continue
            
            # Check for bearish setup
            if i >= 2:
                # Find recent swing high
                recent_swing_high_idx = -1
                for j in range(i-1, max(0, i-20), -1):
                    if not np.isnan(swing_highs[j]):
                        recent_swing_high_idx = j
                        break
                
                if recent_swing_high_idx != -1 and i - recent_swing_high_idx >= 3:
                    # Check for lower high after swing high
                    lower_high_found = False
                    for j in range(recent_swing_high_idx + 1, i):
                        if high[j] < high[recent_swing_high_idx]:
                            lower_high_found = True
                            break
                    
                    if lower_high_found:
                        # Check for break below recent swing low
                        recent_swing_low_idx = -1
                        for j in range(recent_swing_high_idx, i):
                            if not np.isnan(swing_lows[j]):
                                recent_swing_low_idx = j
                                break
                        
                        if (recent_swing_low_idx != -1 and 
                            close[i] < low[recent_swing_low_idx] and
                            vol_spike[i] and 
                            close[i] < ema34_1d_aligned[i]):
                            signals[i] = -0.25
                            position = -1
                            continue
        
        else:
            # Exit: trend exhaustion or 1d EMA34 violation
            if position == 1:
                if close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_123_Reversal_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0