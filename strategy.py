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
    
    # Get 12h data for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR (14-period)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full(len(df_12h), np.nan)
    for i in range(13, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for higher timeframe trend (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 12h data for price structure (Donchian channel breakout)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channel (10-period) on 12h data
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    for i in range(9, len(df_12h)):
        donchian_high[i] = np.max(high_12h[i-9:i+1])
        donchian_low[i] = np.min(low_12h[i-9:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume filter: volume > 1.5x 24-period average (12h)
    vol_ma_24 = np.full(n, np.nan, dtype=np.float64)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h ATR (14), 1d EMA (50), 1w EMA (20), 12h Donchian (10), volume MA (24)
    start_idx = max(14, 50, 20, 10, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        atr_val = atr_14_aligned[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        ema_trend_1w = ema_20_1w_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend alignment: both 1d and 1w EMAs must agree (bullish/bearish)
        bullish_trend = price > ema_trend_1d and price > ema_trend_1w
        bearish_trend = price < ema_trend_1d and price < ema_trend_1w
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish trend alignment + volume spike
            if price > donch_high and bullish_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + bearish trend alignment + volume spike
            elif price < donch_low and bearish_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low (mean reversion) or trend turns bearish or ATR-based stop
            if price <= donch_low or not bullish_trend or price < (donch_high - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high (mean reversion) or trend turns bullish or ATR-based stop
            if price >= donch_high or not bearish_trend or price > (donch_low + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_1wEMA20_Trend_Volume_ATR"
timeframe = "12h"
leverage = 1.0