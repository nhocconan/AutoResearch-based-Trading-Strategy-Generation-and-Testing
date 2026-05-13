#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Filter
# Hypothesis: Enter long when price breaks above Keltner upper band during low volatility, with trend filter from 12h EMA50. Enter short when price breaks below Keltner lower band with trend filter. Uses ATR-based bands for volatility adaptation, reducing false breakouts in choppy markets. Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend). Low frequency due to volatility-based entry and trend confirmation.

name = "4h_Keltner_Breakout_Trend_Filter"
timeframe = "4h"
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

    # Get 12h data for Keltner Channels and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Keltner Channels (20, 2.0) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical Price and ATR
    tp = (high_12h + low_12h + close_12h) / 3
    atr = np.zeros(len(tp))
    tr = np.zeros(len(tp))
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(tp)):
        tr[i] = max(high_12h[i] - low_12h[i], 
                    abs(high_12h[i] - close_12h[i-1]), 
                    abs(low_12h[i] - close_12h[i-1]))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # EMA of Typical Price for middle line
    ema_tp = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper = ema_tp + 2 * atr
    lower = ema_tp - 2 * atr
    
    # 12h EMA50 trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_conf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + uptrend + volume confirmation
            if close[i] > upper_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_conf[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + downtrend + volume confirmation
            elif close[i] < lower_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_conf[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below middle band OR trend reversal
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above middle band OR trend reversal
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals