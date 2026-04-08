# 4h_bollinger_reversion_v2
# Hypothesis: 4h Bollinger Bands mean reversion with volume confirmation and 1d trend filter.
# In ranging markets, price reverts to mean from Bollinger Bands with volume confirmation.
# In trending markets, we filter out counter-trend trades using 1d EMA trend.
# Volume confirms institutional participation at reversal points.
# Target: 20-30 trades/year to avoid overtrading and fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_reversion_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate Bollinger Bands
    basis = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_length, n):
        basis[i] = np.mean(close[i-bb_length:i])
        dev = bb_mult * np.std(close[i-bb_length:i])
        upper[i] = basis[i] + dev
        lower[i] = basis[i] - dev
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 1d trend filter (using close vs 50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA properly
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        # Initialize first value
        ema_50[49] = np.mean(close_1d[:50])
        # Calculate EMA for rest
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(bb_length, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price_position = (close[i] - lower[i]) / (upper[i] - lower[i]) if upper[i] > lower[i] else 0.5
        
        if position == 1:  # Long
            # Exit: price returns to middle or volume drops
            if price_position >= 0.5 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to middle or volume drops
            if price_position <= 0.5 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches lower band with volume and uptrend bias
            if (close[i] <= lower[i] and 
                vol_ratio > 1.5 and 
                close[i] > ema_50_aligned[i]):  # Only long in uptrend
                position = 1
                signals[i] = 0.25
            # Short: price touches upper band with volume and downtrend bias
            elif (close[i] >= upper[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < ema_50_aligned[i]):  # Only short in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals