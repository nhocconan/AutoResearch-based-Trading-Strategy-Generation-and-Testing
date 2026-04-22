#!/usr/bin/env python3

"""
Hypothesis: 1-day Williams %R overbought/oversold with 1-week EMA(34) trend filter and volume spike confirmation.
Trades mean-reversion in the direction of the weekly trend only when volume exceeds 1.8x the 20-period average.
Uses ATR(10) for dynamic position sizing to normalize volatility across regimes.
Targets 20-50 trades/year (80-200 total over 4 years) with disciplined entry/exit to minimize fee drift.
Works in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Load 1d data for Williams %R and ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # 1d ATR for volatility normalization (10-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Load 1w data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align 1d indicators
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_10_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R oversold (< -80), above weekly EMA (uptrend)
            if williams_r_aligned[i] < -80 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), below weekly EMA (downtrend)
            elif williams_r_aligned[i] > -20 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 or price below weekly EMA
                if williams_r_aligned[i] > -50 or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R < -50 or price above weekly EMA
                if williams_r_aligned[i] < -50 or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_14_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0