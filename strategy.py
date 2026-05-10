#!/usr/bin/env python3
# 6H_Williams_Fractal_Breakout_1dTrend
# Hypothesis: Uses Williams fractals (weekly) to identify key support/resistance levels.
# Enters long when price breaks above a weekly bearish fractal with 1d uptrend (close > EMA50).
# Enters short when price breaks below a weekly bullish fractal with 1d downtrend (close < EMA50).
# Uses volume confirmation to avoid false breakouts. Weekly fractals require 2-bar confirmation.
# Works in both bull and bear markets by following the 1d trend direction.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6H_Williams_Fractal_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.zeros(len(high_1w), dtype=bool)
    bullish_fractal = np.zeros(len(low_1w), dtype=bool)
    
    # Williams fractal: need 2 bars on each side for confirmation
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = True
        
        # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Get 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly fractals to 6h with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above weekly bearish fractal in uptrend with volume
            if (bearish_fractal_aligned[i] > 0 and 
                close[i] > bearish_fractal_aligned[i] and  # price above fractal level
                price_above_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly bullish fractal in downtrend with volume
            elif (bullish_fractal_aligned[i] > 0 and 
                  close[i] < bullish_fractal_aligned[i] and  # price below fractal level
                  price_below_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly bullish fractal or trend changes
            if (bullish_fractal_aligned[i] > 0 and 
                close[i] < bullish_fractal_aligned[i]) or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly bearish fractal or trend changes
            if (bearish_fractal_aligned[i] > 0 and 
                close[i] > bearish_fractal_aligned[i]) or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals