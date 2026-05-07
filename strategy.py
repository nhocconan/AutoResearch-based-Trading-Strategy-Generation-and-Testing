#!/usr/bin/env python3
name = "1d_WilliamsFractal_Breakout_1wTrend_PriceAction"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w SMA50 trend filter
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate Williams fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Apply 2-bar delay for fractal confirmation (needs 2 future weekly candles)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Price action: daily close > 2-period high for momentum
    high_2 = pd.Series(high).rolling(window=2, min_periods=2).max().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # ~1 week for 1d to reduce trades
    
    start_idx = max(250, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(high_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = close > sma_50_1w_aligned[i]
        trend_down = close < sma_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bullish fractal break above prior 2-day high in uptrend
            if (bullish_fractal_aligned[i] and 
                close[i] > high_2[i] and 
                trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bearish fractal break below prior 2-day low in downtrend
            elif (bearish_fractal_aligned[i] and 
                  close[i] < low[i-1] and  # simple prior low break
                  trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price closes below 2-day low or trend changes
            if (close[i] < low[i-1] or not trend_up[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above 2-day high or trend changes
            if (close[i] > high_2[i] or not trend_down[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams fractals on weekly timeframe provide institutional-grade
# reversal signals with built-in confirmation. Combining with 1w SMA50 trend filter
# and daily price action (breaking 2-day high/low) creates a robust system that
# works in both bull and bear markets. The weekly timeframe reduces noise and
# whipsaw, while the daily entry timing improves precision. Position size of 0.25
# manages drawdown, and weekly cooldown prevents overtrading. Target: 15-35 trades
# per year (60-140 total over 4 years) to minimize fee drag. Williams fractals
# are particularly effective in cryptocurrency markets due to their clear
# structural breakout signals.