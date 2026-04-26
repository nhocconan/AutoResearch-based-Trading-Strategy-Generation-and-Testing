#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts filtered by 1w EMA50 trend and volume spike.
Camarilla pivot levels provide high-probability intraday support/resistance. In strong 1w trends,
breakouts of R1 (resistance) or S1 (support) have high win rate. Volume confirmation reduces false signals.
Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
Works in both bull and bear markets by following 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla pivot calculation (yesterday's HLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume confirmation: 2.0x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1w EMA (50), 1d data (1), volume median (30)
    start_idx = max(50, 1, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: close breaks above R1, uptrend (close > 1w EMA50), volume spike
            long_signal = (close_val > r1_val) and \
                          (close_val > ema_50_1w_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: close breaks below S1, downtrend (close < 1w EMA50), volume spike
            short_signal = (close_val < s1_val) and \
                           (close_val < ema_50_1w_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: trend reversal (close < 1w EMA50) or price drops below S1 after minimum holding
            if bars_since_entry >= 6 and ((close_val < ema_50_1w_val) or (close_val < s1_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: trend reversal (close > 1w EMA50) or price rises above R1 after minimum holding
            if bars_since_entry >= 6 and ((close_val > ema_50_1w_val) or (close_val > r1_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_v1"
timeframe = "12h"
leverage = 1.0