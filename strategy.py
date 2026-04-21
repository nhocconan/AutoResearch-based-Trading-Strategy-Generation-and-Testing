# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h Volatility Contraction Breakout with 12h EMA50 trend filter and volume spike.
Volatility contraction (low Bollinger Band width) precedes explosive moves. Breakout above/below
Bollinger Bands with volume confirmation and aligned trend captures momentum. Uses 12h EMA50 for
trend filter to avoid counter-trend trades. Designed for fewer trades (~20-40/year) to minimize
fee drag, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Bollinger Bands (20,2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Bollinger Bands: 20-period SMA ± 2*std
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_12h_aligned[i]
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above BB upper, volume spike, uptrend
            if (price_close > bb_upper and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below BB lower, volume spike, downtrend
            elif (price_close < bb_lower and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to SMA20 or trend reversal
            sma_20_val = sma_20_aligned[i] if 'sma_20_aligned' in locals() else None
            if sma_20_val is None:
                sma_20_val = align_htf_to_ltf(prices, df_1d, sma_20)[i]
            
            if position == 1 and (price_close < sma_20_val or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > sma_20_val or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VolatilityContraction_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0