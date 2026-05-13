#!/usr/bin/env python3
# Hypothesis: 6h Williams Fractal breakout with 1d trend filter (EMA34) and volume confirmation (>2.0x 20-bar avg).
# Uses weekly higher timeframe for regime (price > weekly EMA50 = bull regime, < = bear regime) to adjust fractal sensitivity.
# In bull regime: buy breakouts above recent bullish fractal; in bear regime: sell breakdowns below recent bearish fractal.
# Volume confirmation avoids false breakouts. Designed for low frequency (target 50-150 total trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets by adapting to weekly regime and requiring strong volume confirmation.

name = "6h_WilliamsFractal_Breakout_1dEMA34_WeeklyEMA50_Regime_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly EMA50 for regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2 extra delay bars for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after all lookbacks
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime: bull if close > weekly EMA50, bear if close < weekly EMA50
        bull_regime = close[i] > ema_50_1w_aligned[i]
        bear_regime = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # LONG: In bull regime, price breaks above recent bullish fractal with volume spike
            if bull_regime and high[i] > bullish_fractal_aligned[i] and volume[i] > 2.0 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: In bear regime, price breaks below recent bearish fractal with volume spike
            elif bear_regime and low[i] < bearish_fractal_aligned[i] and volume[i] > 2.0 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if price breaks below recent bearish fractal or volume drops significantly
            if low[i] < bearish_fractal_aligned[i] or volume[i] < 0.4 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close if price breaks above recent bullish fractal or volume drops significantly
            if high[i] > bullish_fractal_aligned[i] or volume[i] < 0.4 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals