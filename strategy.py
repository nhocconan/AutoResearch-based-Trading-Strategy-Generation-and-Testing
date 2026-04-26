#!/usr/bin/env python3
"""
1d_KAMA_Regime_ChopFilter_v1
Hypothesis: Daily KAMA direction with weekly trend filter and choppiness regime filter.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Kaufman Adaptive Moving Average (KAMA) adapts to market noise/trend
- Weekly EMA200 ensures trades align with higher timeframe trend
- Choppiness Index (CHOP) filter avoids ranging markets (CHOP > 61.8) and trades only in trending (CHOP < 38.2) or extreme chop (CHOP > 80) for mean reversion
- Long when price > KAMA AND weekly uptrend AND (trending regime OR extreme chop with price < BB lower)
- Short when price < KAMA AND weekly downtrend AND (trending regime OR extreme chop with price > BB upper)
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by combining trend following and mean reversion based on regime
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # KAMA calculation (10, 2, 30)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over window
    
    # Handle array dimensions properly
    change_padded = np.full(n, np.nan)
    change_padded[10:] = change
    
    volatility_padded = np.full(n, np.nan)
    for i in range(1, n):
        start = max(0, i-9)
        volatility_padded[i] = np.sum(np.abs(np.diff(close[start:i+1])))
    
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Bollinger Bands (20, 2) for extreme chop signals
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(1)) over 14 / (max(high)-min(low) over 14)) / log10(14)
    atr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr1 = np.concatenate([[np.nan], atr1])  # align with index
    
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    chop = np.where(range14 != 0, 
                    100 * np.log10(sum_atr14 / range14) / np.log10(14), 
                    50)
    chop = np.nan_to_num(chop, nan=50)
    
    # Align HTF indicators
    kama_aligned = kama  # Already calculated on 1d
    bb_upper_aligned = bb_upper
    bb_lower_aligned = bb_lower
    chop_aligned = chop
    ema200_1w_aligned = ema200_1w_aligned  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for BB, 14 for CHOP, 10 for KAMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime conditions
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Chop regimes
        chop_value = chop_aligned[i]
        trending_regime = chop_value < 38.2
        ranging_regime = chop_value > 61.8
        extreme_chop = chop_value > 80
        
        # Mean reversion signals in extreme chop
        mean_revert_long = extreme_chop and close[i] < bb_lower_aligned[i]
        mean_revert_short = extreme_chop and close[i] > bb_upper_aligned[i]
        
        if position == 0:
            # Long: price > KAMA AND weekly uptrend AND (trending OR mean reversion in extreme chop)
            if price_above_kama and weekly_uptrend and (trending_regime or mean_revert_long):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND weekly downtrend AND (trending OR mean reversion in extreme chop)
            elif price_below_kama and weekly_downtrend and (trending_regime or mean_revert_short):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below KAMA OR weekly trend turns down
            if price_below_kama or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above KAMA OR weekly trend turns up
            if price_above_kama or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Regime_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0