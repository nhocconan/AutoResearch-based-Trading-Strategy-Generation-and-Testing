#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and volume confirmation.
Long when price breaks above R1 in 1w uptrend with volume spike. Short when price breaks below S1 in 1w downtrend with volume spike.
Uses 1d timeframe to target 30-100 trades over 4 years (7-25/year). Camarilla levels provide precise support/resistance.
1w trend filter avoids counter-trend trades in bear markets (like 2022 crash and 2025 bearish BTC/ETH).
Volume spike confirms institutional interest. Discrete position sizing (0.25) minimizes fee churn.
Designed to work in both bull and bear by following 1w trend. Effective in ranging markets due to tight Camarilla levels.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (no extra delay needed as they're based on completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 1 for Camarilla)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend and volume spike
            if (close[i] > R1_aligned[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1w downtrend and volume spike
            elif (close[i] < S1_aligned[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 OR 1w trend changes to downtrend
            if (close[i] < S1_aligned[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 OR 1w trend changes to uptrend
            if (close[i] > R1_aligned[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0