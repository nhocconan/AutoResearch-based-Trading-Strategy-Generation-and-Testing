#!/usr/bin/env python3
"""
4h_Choppiness_Band_Touch_With_Trend_And_Volume
Hypothesis: Price touching Bollinger Bands (20,2) in low volatility (Choppiness > 61.8) with trend confirmation (12h EMA50) and volume spike captures mean-reversion bounces in ranging markets and avoids false signals in trends. Works in sideways/choppy markets (common in 2025) and avoids trend-following whipsaw.
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
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    upper = upper.values
    lower = lower.values
    basis = basis.values
    
    # Choppiness Index (14) - range detection
    def choppiness_index(high, low, close, period=14):
        atr = []
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.abs(high[0] - low[0])], tr])
        for i in range(len(close)):
            if i < period:
                atr.append(np.nan)
            else:
                atr.append(np.sum(tr[i-period+1:i+1]) / period)
        atr = np.array(atr)
        sum_atr = np.nancumsum(atr)  # cumulative sum with NaN handling
        max_high = np.maximum.accumulate(high)
        min_low = np.minimum.accumulate(low)
        range_max_min = max_high - min_low
        chop = 100 * np.log10(sum_atr[period-1:] / range_max_min[period-1:]) / np.log10(period)
        chop_full = np.full_like(close, np.nan, dtype=float)
        chop_full[period-1:] = chop
        return chop_full
    
    chop = choppiness_index(high, low, close, 14)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for BB and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 = ranging market (good for mean reversion)
        if chop[i] <= 61.8:
            # In trending market, stay flat to avoid whipsaw
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB in ranging market with volume spike and uptrend bias
            if (close[i] <= lower[i] and volume_spike[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB in ranging market with volume spike and downtrend bias
            elif (close[i] >= upper[i] and volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle BB or volatility breaks down
            if (close[i] >= basis[i] or chop[i] < 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB or volatility breaks down
            if (close[i] <= basis[i] or chop[i] < 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Band_Touch_With_Trend_And_Volume"
timeframe = "4h"
leverage = 1.0