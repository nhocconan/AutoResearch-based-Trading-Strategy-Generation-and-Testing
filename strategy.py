#!/usr/bin/env python3
# 4h_12h_volatility_breakout_v1
# Strategy: 4h volatility breakout using ATR and Bollinger Bands with 12h trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Volatility expansion (ATR > 1.5x average) combined with price breaking Bollinger Bands (20,2) 
# signals institutional participation. 12h EMA50 filter ensures trades align with higher-timeframe trend.
# Designed for low frequency (~20-40 trades/year) to minimize fee drag in BTC/ETH markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Volatility expansion: current ATR > 1.5x 20-period average ATR
    atr_avg_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_expansion = atr > (1.5 * atr_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: volatility expansion + Bollinger Band break + trend alignment
        if (vol_expansion[i] and close[i] > upper_band[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (vol_expansion[i] and close[i] < lower_band[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: volatility contraction or trend change
        elif position == 1 and (not vol_expansion[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not vol_expansion[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals