#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Williams Fractal breakout and 12h volume confirmation.
# In strong trends, price breaks above/below recent fractal highs/lows (support/resistance).
# Uses 1-day Williams Fractals for key levels, confirmed by 12h volume spike and 12h EMA trend.
# Enters long when price breaks above recent bullish fractal with volume above average and EMA bullish.
# Enters short when price breaks below recent bearish fractal with volume above average and EMA bearish.
# Exits when price returns to the broken fractal level or trend weakens.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_Williams_Fractal_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1-day data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Need 2 extra bars for fractal confirmation (as per Williams Fractal definition)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Get 12h EMA for trend filter (21-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close']
    ema_21 = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    
    # Volume spike detector: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above bullish fractal + volume spike + EMA bullish (price > EMA)
            if (close[i] > bullish_fractal_aligned[i] and
                volume_spike[i] and
                close[i] > ema_21_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below bearish fractal + volume spike + EMA bearish (price < EMA)
            elif (close[i] < bearish_fractal_aligned[i] and
                  volume_spike[i] and
                  close[i] < ema_21_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to bullish fractal level OR trend turns bearish
            if (close[i] <= bullish_fractal_aligned[i] or
                close[i] < ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to bearish fractal level OR trend turns bullish
            if (close[i] >= bearish_fractal_aligned[i] or
                close[i] > ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals