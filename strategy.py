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
    
    # Get 12h data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period Donchian channels on 12h for structure
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    highest_high_12h = np.full(len(high_12h), np.nan)
    lowest_low_12h = np.full(len(low_12h), np.nan)
    
    lookback = 20
    for i in range(lookback, len(high_12h)):
        highest_high_12h[i] = np.max(high_12h[i-lookback:i])
        lowest_low_12h[i] = np.min(low_12h[i-lookback:i])
    
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    
    # Calculate 20-period average volume on 4h for volume confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need at least 20 for EMA, 20 for Donchian, 20 for volume
    start_idx = max(vol_period, lookback) + 20  # Extra buffer for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or
            np.isnan(highest_high_12h_aligned[i]) or
            np.isnan(lowest_low_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine 12h trend
        bullish_trend = ema_12h_aligned[i] > np.mean(ema_12h_aligned[max(0, i-5):i+1])
        bearish_trend = ema_12h_aligned[i] < np.mean(ema_12h_aligned[max(0, i-5):i+1])
        
        # Volume confirmation: spike > 1.8x average
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price breaks above 12h Donchian high in bullish trend with volume
            if bullish_trend and price > highest_high_12h_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: price breaks below 12h Donchian low in bearish trend with volume
            elif bearish_trend and price < lowest_low_12h_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 12h Donchian low or trend turns bearish
            if price < lowest_low_12h_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above 12h Donchian high or trend turns bullish
            if price > highest_high_12h_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_12hDonchian20_EMA20_Trend_Volume"
timeframe = "4h"
leverage = 1.0