#!/usr/bin/env python3
# 6h_12h_maen_reversion_v1
# Strategy: 6h mean reversion using 12h Bollinger Bands and RSI
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In ranging markets, price tends to revert to the mean from Bollinger Band extremes.
# Using 12h Bollinger Bands (20, 2) and 6h RSI (14) to identify overextended conditions.
# Works in both bull and bear markets by fading extremes during ranging periods.
# Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_12h = pd.Series(df_12h['close'])
    sma_20 = close_12h.rolling(window=20, min_periods=20).mean()
    std_20 = close_12h.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 6h
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb.values)
    sma_20_aligned = align_htf_to_ltf(prices, df_12h, sma_20.values)
    
    # Calculate 6h RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion signals
        price = close[i]
        
        # Long when price touches lower BB and RSI is oversold
        long_signal = (price <= lower_bb_aligned[i]) and (rsi[i] < 30)
        
        # Short when price touches upper BB and RSI is overbought
        short_signal = (price >= upper_bb_aligned[i]) and (rsi[i] > 70)
        
        # Exit when price returns to middle band
        exit_long = (position == 1 and price >= sma_20_aligned[i])
        exit_short = (position == -1 and price <= sma_20_aligned[i])
        
        # Track position
        if i == 20:
            position = 0
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals