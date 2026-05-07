# [132912] 12h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1
# Hypothesis: Use 12h timeframe with Camarilla R1/S1 breakouts from prior day, filtered by 1d EMA34 trend and volume spikes.
# This targets fewer trades (~20-50/year) with higher quality: trend alignment + volatility expansion.
# Works in bull (breakouts catch momentum) and bear (mean reversion to pivot point exits).
# Focus on BTC/ETH; avoids overtrading via strict entry conditions.

#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (R1, S1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.382 * camarilla_range
    s1 = prev_close - 0.382 * camarilla_range
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0 x 24-period average (12h * 24 = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure volume MA data
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Break above R1 in 1d uptrend with volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 in 1d downtrend with volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to previous day's close (pivot point)
            prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
            if position == 1 and close[i] < prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals