#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime (ATR-based volatility)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day SMA on daily close for trend filter
    sma_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily ATR and SMA to 4h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    sma_20d_aligned = align_htf_to_ltf(prices, df_1d, sma_20d)
    
    # Calculate 4h ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in HTF indicators
        if np.isnan(atr_14d_aligned[i]) or np.isnan(sma_20d_aligned[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Trend filter: close above/below 20-day SMA
        price = close[i]
        trend_up = price > sma_20d_aligned[i]
        trend_down = price < sma_20d_aligned[i]
        
        if position == 0:
            # Long: price above 20-day SMA + volatility expansion + volume
            if trend_up and atr_4h[i] > atr_14d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below 20-day SMA + volatility expansion + volume
            elif trend_down and atr_4h[i] > atr_14d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility contraction
            if not trend_up or atr_4h[i] <= atr_14d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility contraction
            if not trend_down or atr_4h[i] <= atr_14d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATRExpansion_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0