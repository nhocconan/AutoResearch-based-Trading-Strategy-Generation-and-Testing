#!/usr/bin/env python3
"""
4h_KAMA_Direction_With_Chop_Regime_v1
Hypothesis: 4h Kaufman Adaptive Moving Average (KAMA) trend direction combined with Choppiness Index regime filter.
- KAMA adapts to market noise: follows trend in trending markets, stays flat in choppy regimes
- Choppiness Index (CHOP) > 61.8 indicates ranging market (mean reversion opportunity)
- CHOP < 38.2 indicates trending market (trend following)
- Long when KAMA rising AND CHOP < 38.2 (strong trend)
- Short when KAMA falling AND CHOP < 38.2 (strong trend)
- Uses 1d EMA200 as higher timeframe trend filter to avoid counter-trend trades
- Designed for 20-50 trades/year on 4h timeframe to minimize fee drag
- Works in both bull and bear markets by aligning with 1d trend and using KAMA for adaptive entry/exit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # KAMA calculation (10, 2, 30) - ER = Efficiency Ratio, FAST = 2, SLOW = 30
    # Change = abs(close - close[10])
    change_10 = np.abs(np.subtract(close[10:], close[:-10]))
    # Volatility = sum of abs(close[i] - close[i-1]) for i=1 to 10
    volatility_10 = np.zeros_like(close)
    for i in range(1, n):
        volatility_10[i] = volatility_10[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility_10[i] -= np.abs(close[i-10] - close[i-11])
    # Avoid division by zero
    volatility_10[volatility_10 == 0] = 1e-10
    # ER = change / volatility
    er = np.zeros_like(close)
    er[10:] = change_10 / volatility_10[10:]
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA: first value = close[0], then kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Choppiness Index calculation (14 periods)
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(np.subtract(high[1:], close[:-1]))
    tr3 = np.abs(np.subtract(low[1:], close[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad first element
    tr = np.concatenate([[tr[0]], tr]) if n > 1 else np.array([tr[0]])
    
    # ATR14 = sum of TR over 14 periods
    atr14 = np.zeros_like(close)
    for i in range(1, n):
        atr14[i] = atr14[i-1] + tr[i]
        if i >= 14:
            atr14[i] -= tr[i-14]
    # Avoid division by zero
    atr14[atr14 == 0] = 1e-10
    
    # Highest high and lowest low over 14 periods
    max_high_14 = np.zeros_like(close)
    min_low_14 = np.zeros_like(close)
    for i in range(n):
        start_idx = max(0, i-13)
        max_high_14[i] = np.max(high[start_idx:i+1])
        min_low_14[i] = np.min(low[start_idx:i+1])
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14[range_14 == 0] = 1e-10
    
    # CHOP = 100 * log10(atr14 / range_14) / log10(14)
    chop = 100 * np.log10(atr14 / range_14) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 14 for CHOP)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction (rising/falling)
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Choppiness regime
        chop_trending = chop[i] < 38.2  # Trending market
        chop_ranging = chop[i] > 61.8   # Ranging market
        
        # 1d trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: KAMA rising AND trending market AND daily uptrend
            if kama_rising and chop_trending and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND trending market AND daily downtrend
            elif kama_falling and chop_trending and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falls OR market becomes ranging
            if kama_falling or chop_ranging:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rises OR market becomes ranging
            if kama_rising or chop_ranging:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_With_Chop_Regime_v1"
timeframe = "4h"
leverage = 1.0