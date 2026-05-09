#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Combines daily Camarilla pivot levels with 4h trend and volume confirmation
# to capture reversals in range-bound markets and continuations in trends.
# Works in bull/bear by using pivot levels as dynamic support/resistance.
# Target: 20-40 trades/year on 4h to avoid fee drag.
name = "4h_Camarilla_Pivot_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    # Previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    # Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.5000
    R3 = prev_close + (prev_high - prev_low) * 1.2500
    R2 = prev_close + (prev_high - prev_low) * 1.1666
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    S2 = prev_close - (prev_high - prev_low) * 1.1666
    S3 = prev_close - (prev_high - prev_low) * 1.2500
    S4 = prev_close - (prev_high - prev_low) * 1.5000
    # Align to 4h
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above S1 and EMA50, with volume
            if (price > S1_4h[i] and 
                price > ema50_4h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price below R1 and EMA50, with volume
            elif (price < R1_4h[i] and 
                  price < ema50_4h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses below S1 or EMA50
            if (price < S1_4h[i] or 
                price < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R1 or EMA50
            if (price > R1_4h[i] or 
                price > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals