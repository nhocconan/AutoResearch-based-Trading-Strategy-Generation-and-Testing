#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian breakout
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24h
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or \
           np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        # Session filter: only trade 8-20 UTC
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h Donchian high with volume, above 1d EMA200
            if price > donchian_high_4h_aligned[i] and price > ema_200_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: break below 4h Donchian low with volume, below 1d EMA200
            elif price < donchian_low_4h_aligned[i] and price < ema_200_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 4h Donchian low or trend fails
            if price < donchian_low_4h_aligned[i] or price < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h Donchian high or trend fails
            if price > donchian_high_4h_aligned[i] or price > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hVol_1dEMA200Trend_Session"
timeframe = "1h"
leverage = 1.0