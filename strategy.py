#!/usr/bin/env python3
# 4h_Keltner_Breakout_BollingerReversal_1dTrend_Volume
# Hypothesis: Combine Keltner channel breakouts with Bollinger band reversals on the 4h timeframe, filtered by 1d EMA trend and volume confirmation.
# Keltner channels (ATR-based) capture volatility breakouts, while Bollinger bands identify overbought/oversold conditions.
# In bull markets, we take long breakouts above Keltner upper band when price is near Bollinger lower band (pullback in uptrend).
# In bear markets, we take short breakouts below Keltner lower band when price is near Bollinger upper band (bounce in downtrend).
# The 1d EMA50 filter ensures alignment with the daily trend, reducing false signals.
# Volume confirmation adds conviction to breakout moves.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_Keltner_Breakout_BollingerReversal_1dTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Keltner Channel (20, 2.0) on 4h
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr

    # Calculate Bollinger Bands (20, 2.0) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bollinger_upper = sma_20 + 2.0 * std_20
    bollinger_lower = sma_20 - 2.0 * std_20

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(bollinger_upper[i]) or np.isnan(bollinger_lower[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Keltner upper band + price near Bollinger lower band (oversold pullback) + price above 1d EMA50 (bullish trend) + volume spike
            if (close[i] > keltner_upper[i] and 
                close[i] < bollinger_lower[i] * 1.02 and  # within 2% of Bollinger lower band
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Keltner lower band + price near Bollinger upper band (overbought bounce) + price below 1d EMA50 (bearish trend) + volume spike
            elif (close[i] < keltner_lower[i] and 
                  close[i] > bollinger_upper[i] * 0.98 and  # within 2% of Bollinger upper band
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Keltner lower band or price below 1d EMA50
            if (close[i] < keltner_lower[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Keltner upper band or price above 1d EMA50
            if (close[i] > keltner_upper[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals