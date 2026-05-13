#!/usr/bin/env python3
# Hypothesis: 12h Williams Fractal breakout with 1d trend filter (EMA34) and volume confirmation (1.5x MA20).
# Enters long when price breaks above recent bearish Williams fractal with 1d bullish trend and volume > 1.5x MA20.
# Enters short when price breaks below recent bullish Williams fractal with 1d bearish trend and volume > 1.5x MA20.
# Exits when price crosses the 12h EMA(21) (dynamic stop/reversal).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence.
# Williams Fractals identify potential reversal points; combining with 1d trend ensures alignment with higher timeframe direction.
# Volume confirmation reduces false signals. Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction.

name = "12h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams Fractals on 12h data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Align with 2-bar delay for fractal confirmation (needs 2 future 12h bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # 12h EMA(21) for dynamic exit
    close_series = pd.Series(close)
    ema21_12h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or \
           np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(ema21_12h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above recent bearish Williams fractal with 1d bullish trend and volume spike
            if close[i] > bearish_fractal_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below recent bullish Williams fractal with 1d bearish trend and volume spike
            elif close[i] < bullish_fractal_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 12h EMA(21)
            if close[i] < ema21_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 12h EMA(21)
            if close[i] > ema21_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals