#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above latest bearish fractal AND price > 1w EMA34 (uptrend) AND volume > 1.5x 20-period average
# Short when price breaks below latest bullish fractal AND price < 1w EMA34 (downtrend) AND volume > 1.5x 20-period average
# Exit when price crosses 12h midpoint (average of recent high/low) OR EMA34 filter reverses
# Williams Fractals identify key swing points; 1w EMA34 filters for major trend to avoid counter-trend whipsaws
# Volume confirmation ensures breakout has institutional participation
# Timeframe: 12h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsFractal_Breakout_1wEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams Fractals and midpoint
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Fractals on 12h
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and
            high_12h[i-3] < high_12h[i-1] and
            high_12h[i+1] < high_12h[i-1]):
            bearish_fractal[i-1] = high_12h[i-1]
        
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and
            low_12h[i-3] > low_12h[i-1] and
            low_12h[i+1] > low_12h[i-1]):
            bullish_fractal[i-1] = low_12h[i-1]
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h midpoint for exit (average of recent 12h high/low)
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    rolling_high = high_12h_series.rolling(window=10, min_periods=1).max().values
    rolling_low = low_12h_series.rolling(window=10, min_periods=1).min().values
    midpoint_12h = (rolling_high + rolling_low) / 2.0
    midpoint_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    
    # Volume confirmation on 12h (threshold: 1.5x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above bearish fractal AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint OR price < EMA34 (trend weakening)
            if close[i] < midpoint_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint OR price > EMA34 (trend weakening)
            if close[i] > midpoint_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals