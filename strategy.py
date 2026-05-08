#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CHOP) regime filter combined with 12h Donchian breakout.
# Long when: price breaks above 12h Donchian upper band AND CHOP(14) < 38.2 (trending regime).
# Short when: price breaks below 12h Donchian lower band AND CHOP(14) < 38.2 (trending regime).
# Exit when price crosses back inside the Donchian bands OR CHOP rises above 61.8 (range regime).
# This strategy avoids false breakouts in ranging markets by using CHOP as a regime filter.
# Works in both bull and bear markets by following breakout direction in trending regimes only.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Position size: 0.25 (25% of capital) to manage drawdown.

name = "4h_DonchianBreakout_12h_ChopRegime_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].rolling(window=20, min_periods=20).max().values
    low_12h = df_12h['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_band = align_htf_to_ltf(prices, df_12h, high_12h)
    lower_band = align_htf_to_ltf(prices, df_12h, low_12h)
    
    # Choppiness Index calculation (14-period) on 4h data
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10((hh - ll) / atr) / np.log10(14)
    chop = np.where((hh - ll) > 0, chop_raw, 50.0)  # default to neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Sufficient warmup for Donchian and CHOP
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper band AND trending regime (CHOP < 38.2)
            long_cond = (close[i] > upper_band[i]) and (chop[i] < 38.2)
            # Short conditions: breakout below lower band AND trending regime (CHOP < 38.2)
            short_cond = (close[i] < lower_band[i]) and (chop[i] < 38.2)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower band OR CHOP enters range regime (CHOP > 61.8)
            if close[i] < lower_band[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper band OR CHOP enters range regime (CHOP > 61.8)
            if close[i] > upper_band[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals