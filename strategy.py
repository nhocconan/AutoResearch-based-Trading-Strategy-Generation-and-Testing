#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d trend filter (EMA34) and volume confirmation (>1.8x avg) provides robust directional signals with controlled trade frequency. Long when price > daily EMA34 + breaks above R1 + volume confirmation; short when price < daily EMA34 + breaks below S1 + volume confirmation. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 12h frequency. Daily trend filter avoids whipsaws in counter-trend breakouts.
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
    
    # Get 1d data for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need enough for EMA
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's high, low, close for Camarilla (from completed 1d candles)
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 12h
    prev_high_12h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_12h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_12h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_ = prev_high_12h - prev_low_12h
    # Avoid division by zero
    range_ = np.maximum(range_, 1e-10)
    
    # Camarilla R1, R2, S1, S2 (using standard Camarilla multipliers)
    r1 = prev_close_12h + range_ * 1.1 / 12
    r2 = prev_close_12h + range_ * 1.1 / 6
    s1 = prev_close_12h - range_ * 1.1 / 12
    s2 = prev_close_12h - range_ * 1.1 / 6
    
    # Volume ratio (current / 30-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need previous day data + EMA warmup + volume MA
    start_idx = max(34, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(r2[i]) or
            np.isnan(s1[i]) or np.isnan(s2[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(prev_high_12h[i]) or np.isnan(prev_low_12h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 1.8  # volume at least 1.8x average
        
        if position == 0:
            # Long: price > daily EMA34 + breaks above R1 + volume confirmation
            long_signal = (close[i] > ema_34_1d_aligned[i] and 
                          close[i] > r1[i] and 
                          vol_confirmed)
            
            # Short: price < daily EMA34 + breaks below S1 + volume confirmation
            short_signal = (close[i] < ema_34_1d_aligned[i] and 
                           close[i] < s1[i] and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below daily EMA34 OR breaks below S1 (reversal)
            if close[i] < ema_34_1d_aligned[i] or close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above daily EMA34 OR breaks above R1 (reversal)
            if close[i] > ema_34_1d_aligned[i] or close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0