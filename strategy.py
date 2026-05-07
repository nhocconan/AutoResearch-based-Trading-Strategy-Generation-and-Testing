#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Fractal breakout with daily trend filter and volume confirmation.
# Long when: Price breaks above daily bearish fractal (sell fractal) AND daily EMA34 rising AND volume > 1.5 * EMA20(volume).
# Short when: Price breaks below daily bullish fractal (buy fractal) AND daily EMA34 falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 6-hour EMA20.
# Williams Fractals require 2-bar confirmation on daily timeframe, so we use additional_delay_bars=2.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward fractal breaks and in bear markets via downward fractal breaks.
name = "6h_WilliamsFractal_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20 for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load daily data for Williams Fractals and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: bearish (sell) and bullish (buy)
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n-2] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n-2] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams Fractals need 2 extra daily bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # EMA34 on daily close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > bearish fractal AND EMA34(1d) rising AND volume spike
            long_condition = (close[i] > bearish_fractal_aligned[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Price < bullish fractal AND EMA34(1d) falling AND volume spike
            short_condition = (close[i] < bullish_fractal_aligned[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA20
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA20
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals