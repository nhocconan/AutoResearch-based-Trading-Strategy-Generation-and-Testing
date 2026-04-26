#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter_v2
Hypothesis: On 1d timeframe, enter long when KAMA trend is up AND price > Donchian(20) upper band AND chop regime < 61.8 (trending). Enter short when KAMA trend is down AND price < Donchian(20) lower band AND chop regime < 61.8. Uses discrete sizing (0.0, ±0.30) to limit fee drag. Target: 15-30 trades/year. Works in bull (trend+breakout) and bear (mean reversion in range via chop filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for regime filter (chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # need for chop calculation
        return np.zeros(n)
    
    # Calculate 1w Chop index (Ehler's Chopiness Index)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) sum
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_h = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_h - min_l
    
    # Chop = 100 * log10(atr_sum / range) / log10(14)
    chop_raw = 100 * np.log10(atr_14 / range_14) / np.log10(14)
    chop_1w = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    # Regime: chop < 61.8 = trending (favor breakout), chop > 61.8 = ranging (favor mean reversion)
    # We'll use chop < 61.8 as our regime filter for breakout strategy
    chop_filter = chop_1w < 61.8
    
    # KAMA calculation (close, ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # needs correction
    # Recalculate properly
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            ch = np.abs(close[i] - close[i-10])
            vol = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    er[:10] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Donchian(20) bands
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (~50), Donchian (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # KAMA trend: price above/below KAMA
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        
        if position == 0:
            # Long: KAMA uptrend + price > Donchian high + chop < 61.8 (trending)
            long_signal = kama_uptrend and close[i] > donch_high[i] and chop_filter[i]
            
            # Short: KAMA downtrend + price < Donchian low + chop < 61.8 (trending)
            short_signal = kama_downtrend and close[i] < donch_low[i] and chop_filter[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price falls below KAMA OR chop > 61.8 (range) OR price < Donchian low
            if close[i] < kama[i] or not chop_filter[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price rises above KAMA OR chop > 61.8 (range) OR price > Donchian high
            if close[i] > kama[i] or not chop_filter[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Regime_Filter_v2"
timeframe = "1d"
leverage = 1.0