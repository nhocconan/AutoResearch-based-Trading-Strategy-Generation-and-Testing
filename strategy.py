#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze + 1d VWAP mean reversion
# In low volatility regimes (BB width < 20th percentile), price tends to revert to mean.
# Use 1d VWAP as dynamic mean: long when price < VWAP, short when price > VWAP.
# Exit when BB width expands above 50th percentile (volatility expansion) or price crosses VWAP.
# Works in both bull and bear markets as it captures mean reversion during low volatility periods.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    # Calculate SMA with proper min_periods
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
    
    # Calculate BB bands
    for i in range(bb_period - 1, n):
        if not np.isnan(sma[i]):
            std_dev = np.std(close[i - bb_period + 1:i + 1])
            bb_upper[i] = sma[i] + bb_std * std_dev
            bb_lower[i] = sma[i] - bb_std * std_dev
            bb_width[i] = bb_upper[i] - bb_lower[i]
    
    # Percentile lookback for BB width (use 50 periods lookback)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i - lookback:i + 1]
            valid_vals = window[~np.isnan(window)]
            if len(valid_vals) > 0:
                # Calculate percentile of current value
                bb_width_percentile[i] = (np.sum(valid_vals <= bb_width[i]) / len(valid_vals)) * 100
    
    # 1-day VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Align 1d VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Start after lookback period
        # Skip if required data not ready
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(vwap_aligned[i]) or
            np.isnan(sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bw_percentile = bb_width_percentile[i]
        vwap_price = vwap_aligned[i]
        
        # Entry conditions: low volatility (squeeze) + price deviation from VWAP
        if position == 0:
            # Long: squeeze + price below VWAP
            if bw_percentile < 20 and price < vwap_price:
                position = 1
                signals[i] = position_size
            # Short: squeeze + price above VWAP
            elif bw_percentile < 20 and price > vwap_price:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: volatility expansion or price crosses above VWAP
            if bw_percentile > 50 or price > vwap_price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: volatility expansion or price crosses below VWAP
            if bw_percentile > 50 or price < vwap_price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_BB_Squeeze_VWAP_MeanReversion"
timeframe = "6h"
leverage = 1.0