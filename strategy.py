#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TripleScreen_VolumeBreakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for market regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d weekly trend: 200 EMA
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d ATR for volatility filter
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # 1w average true range for regime filter
    tr1w = df_1w['high'] - df_1w['low']
    tr2w = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3w = abs(df_1w['low'] - df_1w['close'].shift(1))
    trw = pd.concat([tr1w, tr2w, tr3w], axis=1).max(axis=1)
    atr_1w = trw.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # 12h Donchian breakout (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all to 12h
    ema200_1d_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1d_12h[i]) or np.isnan(atr_1d_12h[i]) or 
            np.isnan(atr_1w_12h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_filter = ema200_1d_12h[i]
        volatility_filter = atr_1d_12h[i] / atr_1w_12h[i]  # Relative volatility
        vol_ok = volume[i] > np.median(volume[max(0, i-20):i+1]) * 1.5
        
        if position == 0:
            # Long: Donchian breakout above, bullish trend, high volatility, volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > trend_filter and 
                volatility_filter > 1.2 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below, bearish trend, high volatility, volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < trend_filter and 
                  volatility_filter > 1.2 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown or trend reversal
            if close[i] < donchian_low[i] or close[i] < trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout above or trend reversal
            if close[i] > donchian_high[i] or close[i] > trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals