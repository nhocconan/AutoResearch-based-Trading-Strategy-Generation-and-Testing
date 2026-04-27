#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility measurement
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d SMA(50) for trend filter
    sma_1d_50 = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        sma_1d_50[i] = np.mean(close_1d[i-50:i])
    
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # Calculate 4h Donchian(20) breakout levels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # Donchian needs 20, SMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(sma_1d_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        volatility_filter = atr_1d_aligned[i] > (price * 0.005)
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, volatility filter, and price above SMA
            if (volume_confirmation and 
                volatility_filter and
                price > donchian_high[i] and 
                close[i-1] <= donchian_high[i] and  # just broke out
                price > sma_1d_50_aligned[i]):      # above long-term trend
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, volatility filter, and price below SMA
            elif (volume_confirmation and 
                  volatility_filter and
                  price < donchian_low[i] and 
                  close[i-1] >= donchian_low[i] and  # just broke down
                  price < sma_1d_50_aligned[i]):     # below long-term trend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or volatility drops
            if (price < donchian_low[i] or 
                not volatility_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price breaks above Donchian high or volatility drops
            if (price > donchian_high[i] or 
                not volatility_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Donchian20_VolumeVolatility_SMA50Trend_v1"
timeframe = "4h"
leverage = 1.0