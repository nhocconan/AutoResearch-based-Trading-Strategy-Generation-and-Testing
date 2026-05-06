#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR expansion + 1w Donchian(20) breakout with volume confirmation
# Long when 1d ATR(14) > 1.5 * ATR(50) (volatility expansion) AND price breaks above 1w Donchian(20) upper band
#     AND volume > 1.5 * avg_volume(20) on 6h
# Short when 1d ATR(14) > 1.5 * ATR(50) AND price breaks below 1w Donchian(20) lower band
#     AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses 1w Donchian(20) midpoint (mean reversion to center)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ATR expansion identifies volatility regimes conducive to breakouts
# 1w Donchian(20) provides structural support/resistance levels
# Volume confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "6h_1dATR_Expansion_1wDonchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed 1d bars for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR(14) and ATR(50) for 1d
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR values to 6h timeframe (wait for completed 1d bar)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Get 1w data ONCE before loop for Donchian(20) channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels
    donchian_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_20_1w = (donchian_high_20_1w + donchian_low_20_1w) / 2.0
    
    # Align 1w Donchian values to 6h timeframe (wait for completed 1w bar)
    donchian_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    donchian_mid_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(donchian_high_20_1w_aligned[i]) or np.isnan(donchian_low_20_1w_aligned[i]) or
            np.isnan(donchian_mid_20_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ATR expansion + price breaks above 1w Donchian upper + volume spike, in session
            if (atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i] and 
                close[i] > donchian_high_20_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: ATR expansion + price breaks below 1w Donchian lower + volume spike, in session
            elif (atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i] and 
                  close[i] < donchian_low_20_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals