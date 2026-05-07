#!/usr/bin/env python3
# 12h_WilliamsFractal_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly Williams fractal breaks with 1-week EMA trend filter and volume confirmation.
# Long when price breaks above bearish fractal in uptrend (price > weekly EMA). Short when price breaks below bullish fractal in downtrend (price < weekly EMA).
# Weekly trend filter reduces whipsaws, volume confirmation ensures momentum. Target: 15-30 trades/year on 12h to minimize fee drag.

name = "12h_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get 1w data for Williams fractals and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams fractals on 1w (need 2-bar confirmation after center)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe with 2-bar delay for fractal confirmation
    bearish_fractal_12h = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_12h = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    ema_34_1w_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter on 12h (24-period average = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(bearish_fractal_12h[i]) or np.isnan(bullish_fractal_12h[i]) or 
            np.isnan(ema_34_1w_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > bearish fractal (resistance broken), above weekly EMA (uptrend), volume spike
            if close[i] > bearish_fractal_12h[i] and close[i] > ema_34_1w_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < bullish fractal (support broken), below weekly EMA (downtrend), volume spike
            elif close[i] < bullish_fractal_12h[i] and close[i] < ema_34_1w_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < bullish fractal (support broken) or below weekly EMA (trend change)
            if close[i] < bullish_fractal_12h[i] or close[i] < ema_34_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > bearish fractal (resistance broken) or above weekly EMA (trend change)
            if close[i] > bearish_fractal_12h[i] or close[i] > ema_34_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals