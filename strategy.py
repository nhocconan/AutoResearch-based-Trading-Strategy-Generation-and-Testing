#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1wTrend_v1
Hypothesis: 6h Williams fractal breakout with weekly EMA trend filter. 
- Long when price breaks above latest bullish fractal AND weekly EMA50 uptrend
- Short when price breaks below latest bearish fractal AND weekly EMA50 downtrend
- Uses Williams fractals from completed daily bars (with 2-bar confirmation delay) for structure
- Weekly EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Designed for low frequency (target 12-30 trades/year) to minimize fee drag
- Exit on opposite fractal break or trend reversal
- Novelty: Uses Williams fractals (lagging HTF indicator) with proper delay + weekly trend filter for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for Williams fractals (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals from daily data (needs confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future daily bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter (needs completed weekly candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, plus fractal lookback)
    start_idx = 100  # Conservative warmup for fractals and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams fractal breakout conditions with weekly trend filter
        if position == 0:
            # Long: Price breaks above latest bullish fractal AND weekly uptrend
            if close[i] > bullish_fractal_aligned[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below latest bearish fractal AND weekly downtrend
            elif close[i] < bearish_fractal_aligned[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below latest bearish fractal OR weekly trend turns down
            if close[i] < bearish_fractal_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above latest bullish fractal OR weekly trend turns up
            if close[i] > bullish_fractal_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wTrend_v1"
timeframe = "6h"
leverage = 1.0