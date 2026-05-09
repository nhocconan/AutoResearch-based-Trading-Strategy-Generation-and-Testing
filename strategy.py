#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day ATR-based volatility filter and 12h Donchian channel breakout.
# In low volatility regimes (ATR(30) > ATR(90) indicating declining volatility), price tends to breakout and trend.
# Enters long when price breaks above 12h Donchian upper (20) in low-volatility regime, short when below Donchian lower.
# Uses 1-day ATR ratio as regime filter: ATR(30)/ATR(90) < 1.0 indicates declining volatility favorable for breakouts.
# Exits when volatility regime shifts to high volatility (ATR ratio > 1.0) or price reverses to Donchian midpoint.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_ATR_Volatility_Filter_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(30) and ATR(90)
    atr_30 = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean()
    atr_90 = pd.Series(tr_1d).rolling(window=90, min_periods=90).mean()
    atr_ratio = atr_30 / atr_90  # < 1.0 indicates declining volatility
    
    atr_ratio_values = atr_ratio.values
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_values)
    
    # 12h Donchian channel (20-period) for breakout signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    donchian_upper = high_12h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_12h.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_mid_values = donchian_mid.values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_values)
    
    # Price position relative to Donchian bands
    price_above_upper = close > donchian_upper_aligned
    price_below_lower = close < donchian_lower_aligned
    price_above_mid = close > donchian_mid_aligned
    price_below_mid = close < donchian_mid_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(price_above_upper[i]) or np.isnan(price_below_lower[i]) or
            np.isnan(price_above_mid[i]) or np.isnan(price_below_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: declining volatility (ATR ratio < 1.0) + price breaks above Donchian upper
            if atr_ratio_aligned[i] < 1.0 and price_above_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: declining volatility (ATR ratio < 1.0) + price breaks below Donchian lower
            elif atr_ratio_aligned[i] < 1.0 and price_below_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts (ATR ratio >= 1.0) OR price reverses below Donchian mid
            if atr_ratio_aligned[i] >= 1.0 or price_below_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts (ATR ratio >= 1.0) OR price reverses above Donchian mid
            if atr_ratio_aligned[i] >= 1.0 or price_above_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals