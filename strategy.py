#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams Fractal breakout with 4h trend filter and volume confirmation.
# Uses 4h Williams fractal breakout for signal direction (trend), confirmed by volume spike and 1h price action.
# Designed to work in both bull and bear markets by using fractal breakouts which capture momentum in any regime.
# Timeframe: 1h, with 4h as HTF for trend direction.
# Target: 15-37 trades/year to avoid fee drag.

name = "1h_WilliamsFractal_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Williams fractal and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Williams fractal requires 5 points: high/low of 2 bars before and after
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    bearish_fractal = np.zeros(len(high_4h))
    bullish_fractal = np.zeros(len(low_4h))
    
    for i in range(2, len(high_4h) - 2):
        if (high_4h[i] > high_4h[i-1] and high_4h[i] > high_4h[i-2] and
            high_4h[i] > high_4h[i+1] and high_4h[i] > high_4h[i+2]):
            bearish_fractal[i] = high_4h[i]  # Bearish fractal at high
        if (low_4h[i] < low_4h[i-1] and low_4h[i] < low_4h[i-2] and
            low_4h[i] < low_4h[i+1] and low_4h[i] < low_4h[i+2]):
            bullish_fractal[i] = low_4h[i]   # Bullish fractal at low
    
    # Williams fractals need 2 extra bars for confirmation (see rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike filter: volume > 2.0x 20-period EMA on 1h
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish fractal breakout with volume and above 4h EMA50
            if (bullish_fractal_aligned[i] > 0 and price > bullish_fractal_aligned[i] and
                vol_spike[i] and price > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: bearish fractal breakdown with volume and below 4h EMA50
            elif (bearish_fractal_aligned[i] > 0 and price < bearish_fractal_aligned[i] and
                  vol_spike[i] and price < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below bullish fractal level
            if price < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above bearish fractal level
            if price > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals