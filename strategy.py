#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with 1w EMA50 trend filter.
# Long when price breaks above 1w Donchian upper (20) AND close > 1w EMA50 (uptrend).
# Short when price breaks below 1w Donchian lower (20) AND close < 1w EMA50 (downtrend).
# Exit when price crosses 1w EMA50 (trend reversal).
# Uses discrete position size 0.25. 1w timeframe ensures trading with higher timeframe trend
# to avoid whipsaws in choppy markets. 1d primary timeframe targets 30-100 total trades over
# 4 years (7-25/year) to minimize fee drag. Works in bull markets (capture breakouts) and
# bear markets (capture breakdowns) with trend filter to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1w Indicators: Donchian Channel (20) and EMA50 ===
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    # EMA50 for trend filter
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < EMA50 (trend break)
            if price < ema50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > EMA50 (trend break)
            if price > ema50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > upper (breakout) AND price > EMA50 (uptrend)
            if (price > upper) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < lower (breakdown) AND price < EMA50 (downtrend)
            elif (price < lower) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wDonchian20_EMA50_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0