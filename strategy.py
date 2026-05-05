#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Fractal breakout with 6h EMA34 trend filter and volume confirmation
# Long when price breaks above the most recent 1d bearish fractal AND price > 6h EMA34 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below the most recent 1d bullish fractal AND price < 6h EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses the 6h EMA34 in the opposite direction
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams Fractals provide swing high/low structure from 1d timeframe
# 6h EMA34 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "6h_WilliamsFractal_Breakout_6hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a few daily bars
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on 1d
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-3] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-3] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]
    
    # Align 1d Williams Fractals to 6h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 6h data ONCE before loop for EMA34 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA34
    close_6h_series = pd.Series(close_6h)
    ema34_6h = close_6h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema34_6h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_6h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above the most recent 1d bearish fractal, above 6h EMA34, volume confirmation, in session
            if close[i] > bearish_fractal_aligned[i] and close[i] > ema34_6h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below the most recent 1d bullish fractal, below 6h EMA34, volume confirmation, in session
            elif close[i] < bullish_fractal_aligned[i] and close[i] < ema34_6h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 6h EMA34
            if close[i] < ema34_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 6h EMA34
            if close[i] > ema34_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals