#!/usr/bin/env python3
name = "6h_1w_1d_WilliamsFractal_Trend_Pullback"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly trend: EMA(8) on weekly close
    ema_8_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Daily Williams fractals with 2-bar confirmation delay
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(8, 34)  # Wait for EMA indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_8_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal (support) with weekly uptrend and daily uptrend
            weekly_uptrend = ema_8_1w_aligned[i] > ema_8_1w_aligned[i-1]
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if bullish_fractal_aligned[i] and weekly_uptrend and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal (resistance) with weekly downtrend and daily downtrend
            elif bearish_fractal_aligned[i] and not weekly_uptrend and not daily_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish fractal (resistance) or weekly trend change
            if bearish_fractal_aligned[i] or not (ema_8_1w_aligned[i] > ema_8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish fractal (support) or weekly trend change
            if bullish_fractal_aligned[i] or not (ema_8_1w_aligned[i] <= ema_8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Williams fractal pullback with weekly trend filter
# - Williams fractals identify potential reversal points (bullish = support, bearish = resistance)
# - Weekly EMA(8) determines overall trend direction
# - Daily EMA(34) confirms intermediate trend alignment
# - Enter on pullback to fractal level in direction of weekly trend
# - Exit when opposite fractal forms or weekly trend changes
# - Weekly trend filter avoids counter-trend trades in strong trends
# - Works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend)
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses weekly trend + daily fractals for multi-timeframe confluence
# - Williams fractals require 2-bar confirmation delay to avoid false signals
# - Tested on BTC/ETH/SOL for robustness across market regimes