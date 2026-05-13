#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_12hTrend
# Hypothesis: Enter long when price breaks above Keltner upper band during 12h uptrend with volume confirmation.
# Enter short when price breaks below Keltner lower band during 12h downtrend with volume confirmation.
# Keltner Channel (ATR-based) adapts to volatility better than Bollinger Bands, reducing false breakouts.
# Trend filter from 12h EMA ensures alignment with higher timeframe momentum.
# Volume surge confirms institutional participation. Works in bull (breakouts above upper band in uptrend)
# and bear (breakdowns below lower band in downtrend). Low frequency due to trend and volume filters.

name = "4h_Keltner_Channel_Breakout_12hTrend"
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

    # Get 12h data for Keltner Channel and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Typical Price for ATR calculation
    tp_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    
    # ATR(10) for Keltner Channel
    atr_10 = pd.Series(tp_12h).rolling(window=10, min_periods=10).apply(
        lambda x: np.max(np.abs(np.diff(x, prepend=x[0]))), raw=False
    ).values
    # Simplified ATR: use true range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'][0]], df_12h['close'][:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'][0]], df_12h['close'][:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # EMA(20) for Keltner Channel middle line
    ema20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_keltner = ema20_12h + 2 * atr_10
    lower_keltner = ema20_12h - 2 * atr_10
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_keltner)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Keltner + 12h uptrend + volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Keltner + 12h downtrend + volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA20 (middle) OR trend reversal
            if close[i] < ema20_12h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA20 (middle) OR trend reversal
            if close[i] > ema20_12h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals