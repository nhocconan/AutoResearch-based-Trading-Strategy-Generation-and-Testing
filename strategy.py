#!/usr/bin/env python3
name = "6h_1w_1d_WilliamsFractal_Pullback_Trend"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams fractals (5-bar window: high/low of center bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Weekly trend: EMA(8) on weekly close
    ema_8_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly fractals and trends to 6h timeframe with proper delay
    # Williams fractal needs 2 extra weekly bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(8, 34, 4)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_8_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to bullish fractal support in weekly uptrend with volume
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_8_1w_aligned[i] > ema_8_1w_aligned[i-1]
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if (not np.isnan(bullish_fractal_aligned[i]) and
                low[i] <= bullish_fractal_aligned[i] * 1.002 and  # allow small penetration
                weekly_uptrend and daily_uptrend and vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: pullback to bearish fractal resistance in weekly downtrend with volume
            elif (not np.isnan(bearish_fractal_aligned[i]) and
                  high[i] >= bearish_fractal_aligned[i] * 0.998 and  # allow small penetration
                  not weekly_uptrend and not daily_uptrend and vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below bullish fractal or trend changes
            if (close[i] < bullish_fractal_aligned[i] * 0.995 or
                ema_8_1w_aligned[i] < ema_8_1w_aligned[i-1] or
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above bearish fractal or trend changes
            if (close[i] > bearish_fractal_aligned[i] * 1.005 or
                ema_8_1w_aligned[i] > ema_8_1w_aligned[i-1] or
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Williams fractal pullback with weekly/daily trend and volume confirmation
# - Weekly Williams fractals identify significant swing points from institutional order flow
# - Pullback to bullish fractal support in weekly uptrend = long opportunity
# - Pullback to bearish fractal resistance in weekly downtrend = short opportunity
# - Requires alignment of weekly and daily trends to avoid counter-trend trades
# - Volume spike (2x average) confirms institutional participation at fractal levels
# - Works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend)
# - Williams fractals require 2-bar confirmation to avoid false signals
# - Position size 0.25 targets ~15-35 trades/year, avoiding fee drag on 6s timeframe
# - Uses actual weekly fractals (not resampled) for proper swing point detection