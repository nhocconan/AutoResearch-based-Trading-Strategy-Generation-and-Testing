#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm_v1
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts with weekly trend filter (EMA34) and volume confirmation (>1.8x avg) provides low-frequency, high-conviction signals. Works in bull markets (long when price > weekly EMA34 + R1 breakout) and bear markets (short when price < weekly EMA34 + S1 breakdown). Uses discrete sizing (0.0, ±0.30) to minimize fee churn. Targets 30-100 trades over 4 years (7-25/year) for optimal 1d frequency. Weekly trend avoids whipsaws in counter-trend breakouts, volume confirms institutional interest, Camarilla levels provide precise intraday support/resistance derived from prior session.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla (shifted by 1 for previous completed day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 1d timeframe (no additional delay needed as we use prior day's data)
    prev_high_1d = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_1d = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_1d = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_ = prev_high_1d - prev_low_1d
    # Avoid division by zero
    range_ = np.maximum(range_, 1e-10)
    
    # Camarilla R1, R2, S1, S2
    r1 = prev_close_1d + range_ * 1.1 / 12
    s1 = prev_close_1d - range_ * 1.1 / 12
    
    # Volume ratio (current / 30-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA + previous day data
    start_idx = max(34, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(prev_high_1d[i]) or np.isnan(prev_low_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        vol_confirmed = vol_ratio[i] > 1.8  # volume at least 1.8x average
        
        if position == 0:
            # Long: price > weekly EMA34 + breaks above R1 + volume
            long_signal = (close[i] > ema_34_1w_aligned[i] and 
                          close[i] > r1[i] and 
                          vol_confirmed)
            
            # Short: price < weekly EMA34 + breaks below S1 + volume
            short_signal = (close[i] < ema_34_1w_aligned[i] and 
                           close[i] < s1[i] and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price closes below weekly EMA34 OR breaks below S1 (reversal)
            if close[i] < ema_34_1w_aligned[i] or close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price closes above weekly EMA34 OR breaks above R1 (reversal)
            if close[i] > ema_34_1w_aligned[i] or close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0