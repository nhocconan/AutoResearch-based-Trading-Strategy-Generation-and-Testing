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
    
    # Get daily data for ATR and price
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-period EMA of close for trend
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Align daily ATR and EMA to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema6_6h = align_htf_to_ltf(prices, df_1d, ema6)
    
    # Calculate 6h Bollinger Bands (20, 2) for mean reversion
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_6h[i]) or np.isnan(ema6_6h[i]) or 
            np.isnan(sma20[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma50 = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_6h[i] > atr_ma50[i] if not np.isnan(atr_ma50[i]) else False
        
        # Trend filter: price above EMA6 = bullish, below = bearish
        trend = close[i] > ema6_6h[i]
        
        if position == 0:
            # Mean reversion long: price touches lower BB in uptrend
            if close[i] <= lower[i] and trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price touches upper BB in downtrend
            elif close[i] >= upper[i] and not trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above EMA6 (trend change) or hits upper BB
            if close[i] >= ema6_6h[i] or close[i] >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below EMA6 (trend change) or hits lower BB
            if close[i] <= ema6_6h[i] or close[i] <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_MeanReversion_EMA_Trend_VolFilter"
timeframe = "6h"
leverage = 1.0