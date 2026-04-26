#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_RegimeFilter
Hypothesis: On 6h timeframe, trade breakouts of daily Williams fractals with 1d trend filter and choppiness regime. Go long when price breaks above the most recent daily bearish fractal with 1d uptrend (close > 1d EMA34) and choppy market (CHOP > 61.8). Go short when price breaks below the most recent daily bullish fractal with 1d downtrend (close < 1d EMA34) and choppy market. Exit on opposite fractal break or trend reversal. Designed for 12-37 trades/year on 6h by requiring multi-timeframe alignment and regime filter, reducing fee drag while capturing reversals in ranging markets and continuations in weak trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need at least 10 periods for fractals
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Align Williams fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate choppiness index on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.zeros_like(close_1d)
    
    # True Range
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(tr_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # ATR(14)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(ATR(14)) / (HH(14) - LL(14))) / log10(14)
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_1d = hh_1d - ll_1d
    chop_1d = 100 * np.log10(sum_atr_1d / np.maximum(range_1d, 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need fractals (5 periods) + EMA34 warmup + ATR warmup + CHOP warmup
    start_idx = max(5 + 2, 34, 14, 14)  # 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion at fractals
        choppy_market = chop_1d_aligned[i] > 61.8
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal + 1d uptrend + choppy market
            long_signal = (close[i] > bearish_fractal_aligned[i]) and trend_1d_uptrend and choppy_market
            
            # Short: price breaks below bullish fractal + 1d downtrend + choppy market
            short_signal = (close[i] < bullish_fractal_aligned[i]) and trend_1d_downtrend and choppy_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below bullish fractal OR 1d trend turns down
            if (close[i] < bullish_fractal_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above bearish fractal OR 1d trend turns up
            if (close[i] > bearish_fractal_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0