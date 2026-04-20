#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action strategy using 1-day pivot points with volume confirmation
# Pivot points provide key support/resistance levels that often act as reversal points
# Buying near support (S1) in uptrend and selling near resistance (R1) in downtrend
# Volume > 1.3x average confirms institutional interest at these levels
# Designed for 4h timeframe with selective entries to avoid overtrading
# Target: 25-40 trades per year per symbol (100-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss and filtering
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    # Trend filter: 50-period EMA on 4h
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    is_uptrend = close > ema50
    is_downtrend = close < ema50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long entry: price near S1 support (+0.2% tolerance) + uptrend + volume
            near_s1 = abs(price - s1_aligned[i]) / s1_aligned[i] < 0.002
            long_signal = near_s1 and is_uptrend[i] and vol_filter[i]
            
            # Short entry: price near R1 resistance (-0.2% tolerance) + downtrend + volume
            near_r1 = abs(price - r1_aligned[i]) / r1_aligned[i] < 0.002
            short_signal = near_r1 and is_downtrend[i] and vol_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or price reaches pivot
            stop_loss = entry_price - 2.0 * atr[i]
            at_pivot = abs(price - pivot_aligned[i]) / pivot_aligned[i] < 0.001
            
            if stop_loss <= 0 or price <= stop_loss or at_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or price reaches pivot
            stop_loss = entry_price + 2.0 * atr[i]
            at_pivot = abs(price - pivot_aligned[i]) / pivot_aligned[i] < 0.001
            
            if stop_loss <= 0 or price >= stop_loss or at_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_S1R1_Volume_Trend"
timeframe = "4h"
leverage = 1.0