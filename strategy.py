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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for daily trend (EMA200) - longer term bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 12h data for price structure (Donchian channel breakout)
    df_12h_price = get_htf_data(prices, '12h')
    if len(df_12h_price) < 20:
        return np.zeros(n)
    
    high_12h = df_12h_price['high'].values
    low_12h = df_12h_price['low'].values
    
    # Donchian channel (20-period) on 12h data
    donchian_high = np.full(len(df_12h_price), np.nan)
    donchian_low = np.full(len(df_12h_price), np.nan)
    for i in range(19, len(df_12h_price)):
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h_price, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h_price, donchian_low)
    
    # Volume filter: volume > 1.8x 20-period average (12h)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h EMA (50), 1d EMA (200), 12h Donchian (20), volume MA (20)
    start_idx = max(50, 200, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend_12h = ema_50_12h_aligned[i]
        ema_trend_1d = ema_200_1d_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend alignment: both 12h and 1d EMAs must agree
        bullish_trend = price > ema_trend_12h and price > ema_trend_1d
        bearish_trend = price < ema_trend_12h and price < ema_trend_1d
        
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
            # Exit long: price returns to Donchian low (mean reversion) or trend turns bearish
            if price <= donch_low or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high (mean reversion) or trend turns bullish
            if price >= donch_high or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_12hEMA50_1dEMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0