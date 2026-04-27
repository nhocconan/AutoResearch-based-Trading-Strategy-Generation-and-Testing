#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal + 1d EMA34 trend + volume spike
# Uses 1d Williams fractals for reversal signals (bullish fractal = potential long setup,
# bearish fractal = potential short setup) with 1d EMA34 trend filter to align with higher timeframe momentum.
# Volume confirmation (>2.0x 24-period average) ensures institutional participation.
# Targets 20-30 trades/year to minimize fee decay while capturing high-probability reversals.
# Works in bull/bear markets by using trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest among i-2,i-1,i,i+1,i+2
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest among i-2,i-1,i,i+1,i+2
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra bars for confirmation (bar must close 2 periods after fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Calculate 24-period average volume for spike detection (4h bars in 1d = 6, so 24 = 4d)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: bullish fractal + uptrend + volume spike
            if bullish_fractal_aligned[i] > 0.5 and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish fractal + downtrend + volume spike
            elif bearish_fractal_aligned[i] > 0.5 and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish fractal appears or trend reverses
            if bearish_fractal_aligned[i] > 0.5 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bullish fractal appears or trend reverses
            if bullish_fractal_aligned[i] > 0.5 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsFractal_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0