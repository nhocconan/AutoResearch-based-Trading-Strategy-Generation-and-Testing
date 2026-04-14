#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout (20) with 1d volume confirmation and 1w trend filter.
Long when price breaks above 4h Donchian high (20) during 1w uptrend (price > 1w EMA200) with volume surge (>1.5x 1d volume MA).
Short when price breaks below 4h Donchian low (20) during 1w downtrend (price < 1w EMA200) with volume surge.
Exit when price crosses 4h EMA(40) in opposite direction.
Designed for low turnover: ~20-40 trades/year per symbol.
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
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d volume MA(20)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h Donchian channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA(40) for exit
    close_series = pd.Series(close)
    ema_40 = close_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align 1d volume MA to 4h
    vol_ma_1d_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align 1w EMA200 to 4h
    ema_200_1w_4h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):  # wait for 1w EMA200 warmup
        if position == 0:
            # Long: Donchian breakout + 1w uptrend + volume surge
            if (close[i] > donch_high[i] and 
                close[i] > ema_200_1w_4h[i] and 
                volume[i] > vol_ma_1d_4h[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Donchian breakdown + 1w downtrend + volume surge
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_200_1w_4h[i] and 
                  volume[i] > vol_ma_1d_4h[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price below EMA(40)
            if close[i] < ema_40[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price above EMA(40)
            if close[i] > ema_40[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dVol_1wTrend"
timeframe = "4h"
leverage = 1.0