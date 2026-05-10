#!/usr/bin/env python3
"""
4h_Engulfing_Breakout_1dTrend_Volume
Hypothesis: 4h bullish/bearish engulfing candle at 1d support/resistance (prior day high/low) 
in direction of 1d EMA50 trend, with volume confirmation. Uses price action structure 
and institutional levels for high-probability entries. Works in bull/bear by following 
higher timeframe trend. Target: 20-40 trades/year on 4h to avoid fee drag.
"""

name = "4h_Engulfing_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend and support/resistance
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Prior day high/low as support/resistance
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Shift by 1 to get prior day's levels (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    # First value will be invalid, handled by NaN check
    
    # Align prior day levels to 4h timeframe
    high_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)
    
    # Get price, volume
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) and valid prior day levels
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_1d_prev_aligned[i]) or
            np.isnan(low_1d_prev_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current green candle fully engulfs prior red candle
        bullish_engulf = (close[i] > open_price[i]) and \
                         (open_price[i-1] > close[i-1]) and \
                         (close[i] > open_price[i-1]) and \
                         (open_price[i] < close[i-1])
        
        # Bearish engulfing: current red candle fully engulfs prior green candle
        bearish_engulf = (close[i] < open_price[i]) and \
                         (open_price[i-1] < close[i-1]) and \
                         (close[i] < open_price[i-1]) and \
                         (open_price[i] > close[i-1])
        
        if position == 0:
            # Long: bullish engulf at prior day support (low) AND uptrend AND volume
            if bullish_engulf and low[i] <= low_1d_prev_aligned[i] * 1.001 and \
               close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulf at prior day resistance (high) AND downtrend AND volume
            elif bearish_engulf and high[i] >= high_1d_prev_aligned[i] * 0.999 and \
                 close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulf OR trend turns bearish OR price reaches prior day resistance
            if bearish_engulf or close[i] < ema_50_aligned[i] or high[i] >= high_1d_prev_aligned[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulf OR trend turns bullish OR price reaches prior day support
            if bullish_engulf or close[i] > ema_50_aligned[i] or low[i] <= low_1d_prev_aligned[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals