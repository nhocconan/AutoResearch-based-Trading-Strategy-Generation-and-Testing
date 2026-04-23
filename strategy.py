#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal Breakout with Weekly Trend Filter and Volume Confirmation
- Uses 1-week Williams Fractals to identify significant swing points
- Long: price breaks above most recent weekly bearish fractal with volume confirmation and weekly uptrend
- Short: price breaks below most recent weekly bullish fractal with volume confirmation and weekly downtrend
- Weekly trend: price > weekly EMA34 for uptrend, price < weekly EMA34 for downtrend
- Fractals require 2-bar confirmation (align_htf_to_ltf with additional_delay_bars=2)
- Discrete position sizing (0.25) to minimize fee churn
- Target: 12-35 trades/year (50-140 over 4 years) to avoid fee drag
- Williams Fractals work in both bull and bear markets by identifying key reversal points
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
    volume = prices['volume'].values
    
    # Get 1-week data for Williams Fractals and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Williams Fractals (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Fractals need 2 extra bars for confirmation (rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 34, 20)  # Ensure sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # Get most recent valid fractal levels (not NaN)
        recent_bearish = bearish_fractal_aligned[i]
        recent_bullish = bullish_fractal_aligned[i]
        
        # Long: break above weekly bearish fractal + weekly uptrend + volume spike
        long_signal = False
        if not np.isnan(recent_bearish):
            long_signal = (
                close[i] > recent_bearish and
                weekly_uptrend and
                volume[i] > 2.0 * vol_ma[i]
            )
        
        # Short: break below weekly bullish fractal + weekly downtrend + volume spike
        short_signal = False
        if not np.isnan(recent_bullish):
            short_signal = (
                close[i] < recent_bullish and
                weekly_downtrend and
                volume[i] > 2.0 * vol_ma[i]
            )
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite fractal break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below weekly bullish fractal or trend turns down
                if (not np.isnan(recent_bullish) and close[i] < recent_bullish) or \
                   not weekly_uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above weekly bearish fractal or trend turns up
                if (not np.isnan(recent_bearish) and close[i] > recent_bearish) or \
                   not weekly_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0