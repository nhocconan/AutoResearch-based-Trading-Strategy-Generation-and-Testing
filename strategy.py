#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1 in 1d uptrend with volume > 1.5x 20-period average.
Enters short when price breaks below S1 in 1d downtrend with volume > 1.5x 20-period average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 75-150 trades over 4 years
by requiring confluence of price level, trend, and volume. Works in bull/bear via trend alignment.
"""

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
    
    # Calculate Camarilla pivot levels from previous day
    # Using 4h data, we need daily high/low/close
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r1_1d = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1_1d = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (delayed by 1 day for look-ahead safety)
    r1_4h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    s1_4h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(htf_trend[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long entry: price breaks above R1 in uptrend with volume spike
        if close[i] > r1_4h_aligned[i] and htf_trend[i] == 1 and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        
        # Short entry: price breaks below S1 in downtrend with volume spike
        elif close[i] < s1_4h_aligned[i] and htf_trend[i] == -1 and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        
        # Exit conditions: price returns to Camarilla pivot level (midpoint)
        elif position == 1 and close[i] < (r1_4h_aligned[i] + s1_4h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (r1_4h_aligned[i] + s1_4h_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0