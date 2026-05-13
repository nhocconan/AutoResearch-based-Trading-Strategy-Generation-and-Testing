#!/usr/bin/env python3
"""
1h_4h1d_Pullback_Trend_Momentum
Hypothesis: On 1h, buy pullbacks in uptrend and sell pullbacks in downtrend using 4h EMA21 for trend and 1d EMA34 for higher timeframe filter.
Enter when price returns to EMA21 on 1h with momentum confirmation (price > open) and volume above average.
Exit when price moves away from EMA21 or trend changes.
Designed for low trade frequency (15-30/year) to avoid fee drag.
Works in bull/bear by following higher timeframe trend.
"""

name = "1h_4h1d_Pullback_Trend_Momentum"
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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA21 trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for EMA34 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: price near 4h EMA21 (pullback in uptrend), above 1d EMA34, bullish candle, volume
            if (close[i] > ema21_4h_aligned[i] * 0.995 and  # within 0.5% above EMA21
                close[i] < ema21_4h_aligned[i] * 1.005 and   # within 0.5% below EMA21
                close[i] > ema34_1d_aligned[i] and           # above 1d EMA34 (uptrend filter)
                close[i] > open_[i] and                      # bullish candle
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: price near 4h EMA21 (pullback in downtrend), below 1d EMA34, bearish candle, volume
            elif (close[i] > ema21_4h_aligned[i] * 0.995 and   # within 0.5% above EMA21
                  close[i] < ema21_4h_aligned[i] * 1.005 and   # within 0.5% below EMA21
                  close[i] < ema34_1d_aligned[i] and           # below 1d EMA34 (downtrend filter)
                  close[i] < open_[i] and                      # bearish candle
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price moves away from EMA21 or trend turns down
            if (close[i] < ema21_4h_aligned[i] * 0.98 or   # 2% below EMA21
                close[i] < ema34_1d_aligned[i]):             # broke 1d EMA34
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price moves away from EMA21 or trend turns up
            if (close[i] > ema21_4h_aligned[i] * 1.02 or   # 2% above EMA21
                close[i] > ema34_1d_aligned[i]):             # broke above 1d EMA34
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals