#!/usr/bin/env python3
"""
Hypothesis: Daily Williams %R mean-reversion with weekly trend filter and volume confirmation.
Buy when weekly trend is up (price > 200 EMA), daily Williams %R < -80 (oversold), and volume > 1.5x average.
Sell when weekly trend is down (price < 200 EMA), daily Williams %R > -20 (overbought), and volume > 1.5x average.
Exit when Williams %R reverts to mean (-50) or trend weakens.
Designed for low trade frequency (~10-25/year) to capture mean reversion in strong trends.
Works in both bull and bear markets by aligning with weekly trend direction.
"""

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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_1w_aligned[i]
        williams_r_val = williams_r[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_price = close[i]
        
        if position == 0:
            # Long: weekly uptrend (price > EMA200), daily oversold (W%R < -80), volume confirmation
            if (close_price > ema200_val and williams_r_val < -80 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price < EMA200), daily overbought (W%R > -20), volume confirmation
            elif (close_price < ema200_val and williams_r_val > -20 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R reverts to mean (> -50) OR trend weakens (price < EMA200)
                if williams_r_val > -50 or close_price < ema200_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R reverts to mean (< -50) OR trend weakens (price > EMA200)
                if williams_r_val < -50 or close_price > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsR_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0