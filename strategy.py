#!/usr/bin/env python3
"""
Hypothesis: On the daily timeframe, price tends to respect the 20-day exponential moving average (EMA20) as dynamic support/resistance. 
In trending markets, price pulls back to EMA20 before continuing the trend. In ranging markets, price oscillates around EMA20. 
We combine EMA20 with a volatility filter (ATR-based) and volume confirmation to enter trades with the trend on pullbacks. 
The weekly trend (EMA50) acts as a regime filter to avoid counter-trend trades. Designed for 1d timeframe to capture multi-day swings 
with low frequency (~10-20 trades per year) to minimize fee drag in both bull and bear markets.
"""

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
    
    # Calculate daily EMA20 for dynamic support/resistance
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 20-day volume average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(weekly_ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        # Determine trend based on weekly EMA50
        is_uptrend = price > weekly_ema_50_aligned[i]
        is_downtrend = price < weekly_ema_50_aligned[i]
        
        if position == 0:
            # Long: pullback to EMA20 in uptrend with volume confirmation
            if is_uptrend and price <= ema_20[i] + 0.5 * atr[i] and price >= ema_20[i] - 0.5 * atr[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: pullback to EMA20 in downtrend with volume confirmation
            elif is_downtrend and price >= ema_20[i] - 0.5 * atr[i] and price <= ema_20[i] + 0.5 * atr[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price moves significantly above EMA20 or trend changes
            if price > ema_20[i] + atr[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves significantly below EMA20 or trend changes
            if price < ema_20[i] - atr[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA20_Pullback_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0