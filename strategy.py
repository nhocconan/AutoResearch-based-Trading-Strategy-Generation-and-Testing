#!/usr/bin/env python3
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
    
    # Get daily data for ATR and Bollinger Bands (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for Donchian channel (price channel) - matches our 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian(20) channel
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(30, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_val = atr_1d_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        atr_ma = pd.Series(atr_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1] if i >= 20 else atr_val
        vol_filter = atr_val > atr_ma
        
        # Trend filter: price above/below weekly EMA50
        trend_filter_long = close[i] > ema_50_val
        trend_filter_short = close[i] < ema_50_val
        
        if position == 0:
            # Long: price breaks above upper Donchian with uptrend and high volatility
            if close[i] > upper_val and trend_filter_long and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with downtrend and high volatility
            elif close[i] < lower_val and trend_filter_short and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian or trend turns down
            if close[i] < lower_val or not trend_filter_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian or trend turns up
            if close[i] > upper_val or not trend_filter_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Trend_VolumeFilter_v2"
timeframe = "12h"
leverage = 1.0