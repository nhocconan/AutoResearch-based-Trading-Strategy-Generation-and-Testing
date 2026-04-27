#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily SMA(50) for trend direction
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get 4h data for volume context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily SMA50
        price_above_sma = close[i] > sma_50_1d_aligned[i]
        price_below_sma = close[i] < sma_50_1d_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] < np.median(atr_14_1d_aligned[:i+1]) * 3
        
        # Volume filter: above average volume
        vol_spike = volume[i] > vol_ma_20_4h_aligned[i]
        
        # Long conditions: bullish trend + volatility filter + volume spike
        long_condition = (price_above_sma and vol_filter and vol_spike)
        
        # Short conditions: bearish trend + volatility filter + volume spike
        short_condition = (price_below_sma and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_sma:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_sma:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_SMA50_1D_VolumeFilter_Trend"
timeframe = "4h"
leverage = 1.0