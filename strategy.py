#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractals for swing high/low breakouts with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 1d bullish fractal (swing high) with volume spike and above 1d EMA50.
# Enter short when price breaks below 1d bearish fractal (swing low) with volume spike and below 1d EMA50.
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 12-37 trades/year.
# Williams Fractals provide natural swing points from higher timeframe, volume confirms breakout strength, EMA50 filters intermediate trend.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "12h_WilliamsFractal_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)  # swing high
    bullish_fractal = np.full(n_1d, np.nan)   # swing low
    
    # Williams Fractal: 5-point pattern (requires 2 bars on each side)
    for i in range(2, n_1d - 2):
        # Bearish fractal (swing high): middle bar highest of 5
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (swing low): middle bar lowest of 5
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals require 2 extra bars for confirmation (pattern completes 2 bars after center)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Williams Fractal breakout conditions with volume confirmation
        long_breakout = close[i] > bullish_fractal_aligned[i] and volume_spike[i]
        short_breakout = close[i] < bearish_fractal_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite fractal level or trend reversal
        long_exit = close[i] < bearish_fractal_aligned[i] or below_ema
        short_exit = close[i] > bullish_fractal_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals