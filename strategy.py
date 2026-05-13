#!/usr/bin/env python3
# 12h_Volume_Squeeze_Breakout_Direction_1wTrend
# Hypothesis: Enter long when price breaks above Bollinger upper band during low volatility (squeeze) in the direction of 1w EMA50 trend, confirmed by volume spike.
# Enter short when price breaks below Bollinger lower band during low volatility in the direction of 1w EMA50 trend, confirmed by volume spike.
# Bollinger squeeze identifies low volatility periods preceding breakouts. Volume surge confirms institutional participation.
# Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets.
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to squeeze requirement and strict volume confirmation.

name = "12h_Volume_Squeeze_Breakout_Direction_1wTrend"
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

    # Get weekly data for Bollinger Bands and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / sma20
    # Squeeze: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 2.0 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + squeeze + weekly uptrend + volume spike
            if close[i] > upper_aligned[i] and squeeze_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + squeeze + weekly downtrend + volume spike
            elif close[i] < lower_aligned[i] and squeeze_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below middle band (SMA20) OR trend reversal
            sma20_aligned = align_htf_to_ltf(prices, df_1w, sma20)
            if close[i] < sma20_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above middle band (SMA20) OR trend reversal
            sma20_aligned = align_htf_to_ltf(prices, df_1w, sma20)
            if close[i] > sma20_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals