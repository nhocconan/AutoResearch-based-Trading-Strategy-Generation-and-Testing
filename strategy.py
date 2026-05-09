#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly Bollinger Band squeeze and daily KAMA trend.
# In low volatility regimes (BB width < 15th percentile on weekly), price tends to trend with the daily KAMA.
# Enters long when KAMA turns bullish (price > KAMA and rising) in low-volatility regime,
# short when KAMA turns bearish (price < KAMA and falling) in low-volatility regime.
# Uses 1-day ATR filter to avoid choppy markets: only trade when ATR(14) < 50th percentile of ATR(50).
# Exits when volatility regime shifts to high volatility or KAMA direction changes.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WeeklyBB_Squeeze_DailyKAMA_Trend"
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
    
    # Calculate weekly Bollinger Bands (20, 2) for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    sma_20w = close_1w.rolling(window=20, min_periods=20).mean()
    std_20w = close_1w.rolling(window=20, min_periods=20).std()
    upper_bbw = sma_20w + 2 * std_20w
    lower_bbw = sma_20w - 2 * std_20w
    bb_width_w = upper_bbw - lower_bbw
    
    # Bollinger Band squeeze: low volatility when BB width < 15th percentile
    bb_width_percentile_w = bb_width_w.rolling(window=50, min_periods=50).quantile(0.15)
    bb_squeeze_w = bb_width_w < bb_width_percentile_w
    bb_squeeze_w_values = bb_squeeze_w.values
    bb_squeeze_w_aligned = align_htf_to_ltf(prices, df_1w, bb_squeeze_w_values)
    
    # Calculate daily KAMA (10, 2, 30)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    # Efficiency Ratio
    change = abs(close_1d - close_1d.shift(10))
    volatility = abs(close_1d - close_1d.shift(1)).rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    kama_values = kama
    kama_rising_values = kama_rising.values
    kama_falling_values = kama_falling.values
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising_values)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling_values)
    
    # Price position relative to KAMA
    price_above_kama = close > kama_aligned
    price_below_kama = close < kama_aligned
    
    # Daily ATR filter: avoid choppy markets
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    atr_50 = tr.rolling(window=50, min_periods=50).mean()
    atr_ratio = atr_14 / atr_50
    atr_filter = atr_ratio < 0.5  # Only trade when short-term ATR < 50% of long-term ATR
    atr_filter_values = atr_filter.values
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_squeeze_w_aligned[i]) or
            np.isnan(kama_aligned[i]) or
            np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or
            np.isnan(price_above_kama[i]) or np.isnan(price_below_kama[i]) or
            np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility (weekly BB squeeze) + price above KAMA + KAMA rising + ATR filter
            if (bb_squeeze_w_aligned[i] and price_above_kama[i] and 
                kama_rising_aligned[i] and atr_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility (weekly BB squeeze) + price below KAMA + KAMA falling + ATR filter
            elif (bb_squeeze_w_aligned[i] and price_below_kama[i] and 
                  kama_falling_aligned[i] and atr_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR KAMA turns bearish OR ATR filter fails
            if (not bb_squeeze_w_aligned[i]) or (not kama_rising_aligned[i]) or (not atr_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR KAMA turns bullish OR ATR filter fails
            if (not bb_squeeze_w_aligned[i]) or (not kama_falling_aligned[i]) or (not atr_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals