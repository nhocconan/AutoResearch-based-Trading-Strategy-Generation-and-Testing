#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Williams Fractals from prior completed 1d bar for structure (breakout = momentum)
# 1w EMA34 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias, more robust across regimes)

name = "1d_WilliamsFractal_Breakout_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and volume EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to price levels (use the fractal high/low values)
    bearish_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars must close)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume (using 1d volume)
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal AND price > 1w EMA34 AND volume spike
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bearish fractal AND price < 1w EMA34 AND volume spike
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to bearish fractal OR price crosses below 1w EMA34
            if close[i] < bearish_fractal_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to bullish fractal OR price crosses above 1w EMA34
            if close[i] > bullish_fractal_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals