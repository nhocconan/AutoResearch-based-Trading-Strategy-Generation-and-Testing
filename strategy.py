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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly SMA(10) for trend direction
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly volume MA(10) for volume filter
    vol_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_10_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(vol_ma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA10
        price_above_sma = close[i] > sma_10_1w_aligned[i]
        price_below_sma = close[i] < sma_10_1w_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1w_aligned[i] > 0 and atr_14_1w_aligned[i] < np.median(atr_14_1w_aligned[:i+1]) * 2
        
        # Volume filter: above average weekly volume
        vol_spike = volume[i] > vol_ma_10_1w_aligned[i]
        
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

name = "12h_WeeklySMA10_VolumeFilter_Session"
timeframe = "12h"
leverage = 1.0