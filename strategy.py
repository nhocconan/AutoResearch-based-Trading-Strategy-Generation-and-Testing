#!/usr/bin/env python3
# Hypothesis: 4h Williams Fractal breakout with 1-day EMA trend filter and volume confirmation.
# Williams Fractals identify potential reversal points; breaks above/below recent fractals with
# trend alignment capture momentum moves. Volume filter ensures breakout legitimacy.
# Works in bull markets via upside breaks and bear via downside breaks of fractal structure.
# Target: 20-50 trades/year with size 0.25.

name = "4h_WilliamsFractal_EMA1d_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1-day close
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1-day data (requires 5-point pattern)
    # Bearish fractal: high[n-2] < high[n] > high[n+2] and low[n-2] < low[n] > low[n+2]
    # Bullish fractal: high[n-2] > high[n] < high[n+2] and low[n-2] > low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i] > high_1d[i+2] and
            low_1d[i-2] < low_1d[i] and low_1d[i] > low_1d[i+2]):
            bearish_fractal[i] = True
        if (high_1d[i-2] > high_1d[i] and high_1d[i] < high_1d[i+2] and
            low_1d[i-2] > low_1d[i] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra bars for confirmation (pattern completes 2 bars after center)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish fractal breakout + price above 1d EMA + volume confirmation
            if bullish_fractal_aligned[i] and close[i] > ema_34_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal breakout + price below 1d EMA + volume confirmation
            elif bearish_fractal_aligned[i] and close[i] < ema_34_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA OR bearish fractal appears
            if close[i] < ema_34_aligned[i] or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA OR bullish fractal appears
            if close[i] > ema_34_aligned[i] or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals