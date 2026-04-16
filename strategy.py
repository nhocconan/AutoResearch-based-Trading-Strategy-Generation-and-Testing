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
    
    # === 1d data (HTF for trend and volatility) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data (HTF for regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate ATR(14) on 1d for volatility regime ===
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Calculate 20-period SMA on 1w for trend regime ===
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # === Align HTF indicators to 6h ===
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # === 6h Bollinger Bands (20, 2) ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_val = atr_14_aligned[i]
        sma_1w_val = sma_20_1w_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        
        # === REGIME FILTER ===
        # High volatility regime: ATR > 1.5 * ATR_MA (using 50-period MA of ATR)
        atr_ma_50 = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma_50[i]):
            signals[i] = 0.0
            position = 0
            continue
        high_vol = atr_val > 1.5 * atr_ma_50[i]
        
        # Trending regime: price > 1w SMA20
        uptrend = price > sma_1w_val
        downtrend = price < sma_1w_val
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches lower BB or volatility drops
            if price < lower or not high_vol:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches upper BB or volatility drops
            if price > upper or not high_vol:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and high_vol:
            # LONG: In uptrend, price breaks above upper BB
            if uptrend and price > upper:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: In downtrend, price breaks below lower BB
            elif downtrend and price < lower:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Bollinger_Breakout_VolRegime_TrendFilter"
timeframe = "6h"
leverage = 1.0