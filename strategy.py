#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Fractal breakout with daily EMA34 trend filter and volume spike.
Long when bullish fractal breaks with price above daily EMA34 and volume spike.
Short when bearish fractal breaks with price below daily EMA34 and volume spike.
Uses 1-day timeframe for trend and fractal confirmation to avoid false breakouts.
Designed for low trade frequency (15-35 trades/year) requiring fractal breakout,
trend alignment, and volume confirmation. Works in both bull and bear markets by
following the daily EMA34 trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Fractals on daily
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Need 2 extra bars for fractal confirmation (requires 2 candles after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: bullish fractal break + price above daily EMA34 + volume spike
            if bullish_fractal_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal break + price below daily EMA34 + volume spike
            elif bearish_fractal_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: fractal in opposite direction or price crosses EMA34
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish fractal or price below EMA34
                if bearish_fractal_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish fractal or price above EMA34
                if bullish_fractal_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0