#!/usr/bin/env python3
"""
1d_Equity_Curve_Momentum
Hypothesis: Daily equity curve momentum (price above 20-day EMA) combined with volume surge and weekly trend alignment captures institutional participation. Works in bull via trend-following and in bear via mean-reversion off weekly extremes when equity curve turns up from oversold.
"""

name = "1d_Equity_Curve_Momentum"
timeframe = "1d"
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
    
    # Daily EMA20 for equity curve
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly trend: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: price above daily EMA20, volume surge, and above weekly EMA50 (bullish alignment)
            if (close[i] > ema20[i] and 
                volume_surge[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price below daily EMA20, volume surge, and below weekly EMA50 (bearish alignment)
            elif (close[i] < ema20[i] and 
                  volume_surge[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below daily EMA20 or weekly trend turns down
            if (close[i] < ema20[i] or 
                close[i] < trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above daily EMA20 or weekly trend turns up
            if (close[i] > ema20[i] or 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals