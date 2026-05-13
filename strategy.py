#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter (EMA34) and volume confirmation.
# Enters long when Lips > Teeth > Jaw (bullish alignment) and price > Lips with 1d bullish trend (close > EMA34) and volume > 1.5x MA20.
# Enters short when Lips < Teeth < Jaw (bearish alignment) and price < Lips with 1d bearish trend (close < EMA34) and volume > 1.5x MA20.
# Exits when Alligator alignment breaks (Lips crosses Teeth or Jaw) or when price crosses opposite Jaw level.
# Uses discrete position sizing (0.25) to minimize fee drag.
# Designed for low trade frequency (~12-37/year) by requiring Alligator alignment + 1d trend + volume confirmation.
# Williams Alligator identifies trending vs ranging markets: aligned = trending, intertwined = ranging.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Alligator provides precise entry/exit signals with volume confirmation reducing false signals.

name = "12h_WilliamsAlligator_Alignment_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    # We'll use EMA as proxy for SMMA since they're similar in behavior
    close_series = pd.Series(close)
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (13)
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values   # Teeth (8)
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips (5)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        if position == 0:
            # LONG: Bullish Alligator alignment + price above Lips + 1d bullish trend + volume spike
            if bullish_alignment and close[i] > lips[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment + price below Lips + 1d bearish trend + volume spike
            elif bearish_alignment and close[i] < lips[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR price crosses below Jaw (strong reversal signal)
            if not bullish_alignment or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR price crosses above Jaw (strong reversal signal)
            if not bearish_alignment or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals