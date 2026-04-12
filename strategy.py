#!/usr/bin/env python3
"""
6h_1d_Williams_Fractal_Trend_v1
Hypothesis: Use daily Williams fractals (with 2-bar confirmation) to identify swing points,
then trade in direction of 6h EMA(21) trend. Long when bullish fractal confirmed and price above EMA,
short when bearish fractal confirmed and price below EMA. Uses volume confirmation to avoid false breaks.
Williams fractals work well in trending markets (2021-2024) and provide clear swing points for trend continuation.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Works in bull via buying pullbacks to EMA, in bear via selling rallies to EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

name = "6h_1d_Williams_Fractal_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals (requires 5-bar window: 2 left, 1 center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Williams fractals need 2 extra bars for confirmation (right side of pattern)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 6h EMA(21) for trend filter
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend and fractal conditions
        uptrend = close[i] > ema_21[i]
        downtrend = close[i] < ema_21[i]
        
        bullish_fractal_signal = bullish_fractal_aligned[i] == 1
        bearish_fractal_signal = bearish_fractal_aligned[i] == 1
        
        # Entry conditions with volume filter
        long_entry = uptrend and bullish_fractal_signal and vol_ratio[i] > 1.5
        short_entry = downtrend and bearish_fractal_signal and vol_ratio[i] > 1.5
        
        # Exit conditions: trend reversal
        long_exit = not uptrend  # price below EMA
        short_exit = not downtrend  # price above EMA
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals