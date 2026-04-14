#!/usr/bin/env python3
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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly EMA10 for trend (1w)
    ema10_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[:10])
        for i in range(10, len(df_1w)):
            ema10_1w[i] = (close_1w[i] * 2 + ema10_1w[i-1] * 8) / 10
    
    ema10_1d = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate weekly EMA60 for trend filter (1w)
    ema60_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 60:
        ema60_1w[59] = np.mean(close_1w[:60])
        for i in range(60, len(df_1w)):
            ema60_1w[i] = (close_1w[i] * 2 + ema60_1w[i-1] * 58) / 60
    
    ema60_1d = align_htf_to_ltf(prices, df_1w, ema60_1w)
    
    # Calculate daily Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily EMA20 for dynamic exit
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        for i in range(20, n):
            ema20[i] = (close[i] * 2 + ema20[i-1] * 18) / 20
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d[i]) or
            np.isnan(ema10_1d[i]) or
            np.isnan(ema60_1d[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        vol_ma_20 = np.mean(volume[i-19:i+1]) if i >= 19 else 0
        if vol_ma_20 <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20
        
        # Volume threshold: require significant spike
        vol_threshold = 1.5
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume confirmation and weekly EMA10 > EMA60
            if close[i] > donch_high[i] and volume_ratio > vol_threshold and ema10_1d[i] > ema60_1d[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low with volume confirmation and weekly EMA10 < EMA60
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold and ema10_1d[i] < ema60_1d[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily EMA20
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily EMA20
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_EMA10_EMA60_Volume"
timeframe = "1d"
leverage = 1.0