#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Uses 1h only for entry timing, 4h for trend direction. Designed for BTC/ETH robustness in both bull/bear regimes: breakouts from Camarilla levels capture volatility expansion, EMA50 filter ensures trend alignment, volume confirms institutional participation. Session filter (08-20 UTC) reduces noise. Targets 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_VolumeSession_v1"
timeframe = "1h"
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
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels (R1, S1) for breakout entries
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above Camarilla R1, price > 4h EMA50, volume spike (>1.5x avg)
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Close breaks below Camarilla S1, price < 4h EMA50, volume spike (>1.5x avg)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S1 (mean reversion) OR close < 4h EMA50 (trend break)
            if (close[i] < camarilla_s1_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R1 (mean reversion) OR close > 4h EMA50 (trend break)
            if (close[i] > camarilla_r1_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals