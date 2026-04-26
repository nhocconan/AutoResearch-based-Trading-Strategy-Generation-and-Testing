#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 in 12h uptrend with volume spike.
Short when price breaks below S1 in 12h downtrend with volume spike.
Uses Camarilla pivot levels from daily OHLC for robust support/resistance.
Discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year on 4h.
Works in bull/bear by following 12h trend. Camarilla levels act as magnetic pivot points.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.25 * (high - low)
    # R2 = close + 1.166 * (high - low)
    # R1 = close + 1.083 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 1.083 * (high - low)
    # S2 = close - 1.166 * (high - low)
    # S3 = close - 1.25 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.083 * range_1d
    camarilla_s1 = close_1d - 1.083 * range_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align all HTF data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
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
                uptrend_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 12h downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  downtrend_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 OR 12h trend changes to downtrend
            if (close[i] < camarilla_s1_aligned[i] or not uptrend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 OR 12h trend changes to uptrend
            if (close[i] > camarilla_r1_aligned[i] or not downtrend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0