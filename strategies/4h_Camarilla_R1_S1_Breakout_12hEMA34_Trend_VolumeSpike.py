#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation (>2.0x 20-period MA).
Long when price breaks above R1 in 12h uptrend with volume spike. Short when price breaks below S1 in 12h downtrend with volume spike.
Camarilla R1/S1 represent the first support/resistance levels (R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12) providing early entry with trend alignment.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-50 trades/year on 4h timeframe.
Works in both bull and bear markets by following the 12h trend. Weekly trend filters out counter-trend noise.
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
    
    # Get 1d data for Camarilla pivot calculation (prior completed daily candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d candle (avoid look-ahead)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from prior completed 1d candle
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (they change only when new 1d candle forms)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    uptrend_12h = close > ema_34_12h_aligned
    downtrend_12h = close < ema_34_12h_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 12h uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                uptrend_12h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 12h downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  downtrend_12h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 (breakdown) OR 12h trend changes to downtrend
            if (close[i] < camarilla_s1_aligned[i] or not uptrend_12h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 (breakout) OR 12h trend changes to uptrend
            if (close[i] > camarilla_r1_aligned[i] or not downtrend_12h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0