#!/usr/bin/env python3
# 12h_Keltner_Channel_Breakout_1dTrend_VolumeFilter
# Hypothesis: Enter long when price breaks above Keltner upper band (EMA20 + 2*ATR) during alignment with 1d EMA50 uptrend, confirmed by volume spike.
# Enter short when price breaks below Keltner lower band (EMA20 - 2*ATR) during alignment with 1d EMA50 downtrend, confirmed by volume spike.
# Keltner Channels adapt to volatility via ATR, providing dynamic support/resistance.
# Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume spike confirms institutional participation in the breakout.
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to ATR-based bands and strict volume confirmation.

name = "12h_Keltner_Channel_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for ATR (needed for Keltner Channels)
    df_1d = get_htf_data(prices, '1d')
    
    # True Range calculation for ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # first value
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner Channels (EMA20, 2*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + daily uptrend + volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + daily downtrend + volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA20 OR trend reversal
            ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
            if close[i] < ema20_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA20 OR trend reversal
            ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
            if close[i] > ema20_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals