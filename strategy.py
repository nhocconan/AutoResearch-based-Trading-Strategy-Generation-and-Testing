#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Donchian channel (price channel)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) channel
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 4h volume for confirmation
    volume_4h = df_4h['volume'].values
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_val = atr_1d_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_val = volume_4h_aligned[i]
        
        # Volume filter: current volume > 20-period average (volume confirmation)
        vol_ma = pd.Series(volume_4h_aligned[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1] if i >= 20 else vol_val
        vol_filter = vol_val > vol_ma
        
        # Trend filter: price > daily EMA50 for long, price < daily EMA50 for short
        price_above_ema = close[i] > ema_50_val
        price_below_ema = close[i] < ema_50_val
        
        if position == 0:
            # Long: price breaks above upper Donchian with price > daily EMA50 and volume confirmation
            if close[i] > upper_val and price_above_ema and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with price < daily EMA50 and volume confirmation
            elif close[i] < lower_val and price_below_ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian or price crosses below daily EMA50
            if close[i] < lower_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian or price crosses above daily EMA50
            if close[i] > upper_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_DailyEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0