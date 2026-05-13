#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Filter
Hypothesis: Camarilla pivot points on 1h with 4h trend filter provide high-probability breakout entries.
In bull markets, buy R1 breaks; in bear markets, sell S1 breaks. Volume confirmation filters false breaks.
Designed for low trade frequency (15-35/year) to minimize fee drag. Works in both bull and bear markets
by using 4h trend to determine direction and 1h Camarilla levels for precise entries.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (using previous bar's OHLC)
    # Camarilla formulas: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_width = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 4h trend filter (EMA 50)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and 4h uptrend
            if close[i] > r1[i] and volume_confirm[i] and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and 4h downtrend
            elif close[i] < s1[i] and volume_confirm[i] and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal) or 4h trend turns down
            if close[i] < s1[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal) or 4h trend turns up
            if close[i] > r1[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals