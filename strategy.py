#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Long: 6h close > H4 (Camarilla resistance) AND 1d volume > 1.5 * 20-period average volume
    # Short: 6h close < L4 (Camarilla support) AND 1d volume > 1.5 * 20-period average volume
    # Exit: Opposite Camarilla level touch (H3/L3) or volume drops below average
    # Using Camarilla from 1d for structure, 6h for execution, 1d volume for confirmation
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    r = prev_high - prev_low
    h4 = prev_close + 1.5 * r
    h3 = prev_close + 1.1 * r
    l3 = prev_close - 1.1 * r
    l4 = prev_close - 1.5 * r
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma_20  # current volume relative to average
    
    # Align 1d indicators to 6h (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: significant volume spike
        volume_confirmed = vol_ratio_aligned[i] > 1.5
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h4_aligned[i] and volume_confirmed
        short_breakout = close[i] < l4_aligned[i] and volume_confirmed
        
        # Exit conditions: retracement to H3/L3 or volume normalization
        long_exit = close[i] < h3_aligned[i] or vol_ratio_aligned[i] < 1.0
        short_exit = close[i] > l3_aligned[i] or vol_ratio_aligned[i] < 1.0
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0