#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray components (13-period EMA)
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6t
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Daily EMA13 for trend filter
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Volume spike detection (20-period SMA on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive AND bear power negative (bullish divergence) + price above EMA13 + volume spike
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                close[i] > ema_13_1d_aligned[i] and volume[i] > vol_ma_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive AND bull power negative (bearish divergence) + price below EMA13 + volume spike
            elif (bear_power_6h[i] > 0 and bull_power_6h[i] < 0 and 
                  close[i] < ema_13_1d_aligned[i] and volume[i] > vol_ma_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bull power turns negative OR price below EMA13
            if bull_power_6h[i] <= 0 or close[i] < ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bear power turns negative OR price above EMA13
            if bear_power_6h[i] <= 0 or close[i] > ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear power) with daily trend filter and volume confirmation
# - Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on daily)
# - Long when Bull Power > 0 AND Bear Power < 0 (bullish divergence) + price above EMA13 + volume spike
# - Short when Bear Power > 0 AND Bull Power < 0 (bearish divergence) + price below EMA13 + volume spike
# - Daily EMA13 trend filter ensures alignment with higher timeframe trend
# - Volume spike (2x 20-period average) reduces false signals
# - Exit when power diverges or price crosses EMA13
# - Works in bull markets (bull power dominance) and bear markets (bear power dominance)
# - Divergence between powers indicates strong directional momentum
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Position size 0.25 balances return potential with drawdown control
# - Novel application: Elder Ray divergence (not just single power) + trend + volume filter
# - Avoids overtrading by requiring confluence of 4 conditions for entry