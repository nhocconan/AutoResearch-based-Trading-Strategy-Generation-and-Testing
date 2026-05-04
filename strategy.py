#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout + 1w Trend Filter + Volume Spike
# Williams fractals identify key swing points. Breakout above bearish fractal (resistance) or below bullish fractal (support) with volume confirms momentum.
# 1-week EMA34 ensures alignment with higher timeframe trend - only take longs in uptrend, shorts in downtrend.
# Designed for 12-37 trades/year on 12h to minimize fee drag. Works in bull markets via breakouts above resistance in uptrend and in bear markets via breakdowns below support in downtrend.

name = "12h_WilliamsFractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-1] > high_1d[i-3] and 
            high_1d[i-1] > high_1d[i+1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Place at the center bar
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i-3] and 
            low_1d[i-1] < low_1d[i+1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Place at the center bar
    
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal (resistance) AND above 1w EMA34 (uptrend) AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bullish fractal (support) AND below 1w EMA34 (downtrend) AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below bullish fractal (support) OR below 1w EMA34
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above bearish fractal (resistance) OR above 1w EMA34
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals