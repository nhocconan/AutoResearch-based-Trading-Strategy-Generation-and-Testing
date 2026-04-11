#!/usr/bin/env python3
# 4h_1d_williams_fractal_breakout_v1
# Strategy: 4h Williams Fractal breakout with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Williams Fractals identify significant swing points. Breakouts above/below recent fractals with volume and 1d trend alignment capture momentum. Works in bull (breakouts up) and bear (breakouts down). Low frequency due to fractal rarity and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "4h_1d_williams_fractal_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_ata(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals on 1d (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Additional 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 4-period high/low for breakout levels
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(4, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(high_4[i]) or 
            np.isnan(low_4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout levels from recent swing points
        # For longs: break above recent bullish fractal OR recent 4-period high
        long_break = (
            (bullish_fractal_aligned[i] > 0 and close[i] > bullish_fractal_aligned[i]) or
            close[i] > high_4[i]
        )
        # For shorts: break below recent bearish fractal OR recent 4-period low
        short_break = (
            (bearish_fractal_aligned[i] > 0 and close[i] < bearish_fractal_aligned[i]) or
            close[i] < low_4[i]
        )
        
        # Entry logic: breakout + volume + trend alignment
        if long_break and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_break and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or trend change
        elif position == 1 and (short_break or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_break or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals