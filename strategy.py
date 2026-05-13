#!/usr/bin/env python3
"""
4h_Aggressive_Trend_Scalper_12hTrend_Volume
Hypothesis: Aggressive trend-following on 4h using 12h EMA trend filter, volume confirmation, and price breakout from recent highs/lows. Designed to capture strong momentum moves in both bull and bear markets with tight risk control. Target: 15-25 trades/year per symbol.
"""

name = "4h_Aggressive_Trend_Scalper_12hTrend_Volume"
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
    
    # Get 4h data for range calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4-period high/low for breakout (16 periods = 1 day on 4h)
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above recent high with volume and uptrend
            if (close[i] > high_4[i] and 
                volume_filter[i] and 
                close[i] > ema20_12h_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Breakdown below recent low with volume and downtrend
            elif (close[i] < low_4[i] and 
                  volume_filter[i] and 
                  close[i] < ema20_12h_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below recent low or trend reverses
            if (close[i] < low_4[i]) or \
               (close[i] < ema20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above recent high or trend reverses
            if (close[i] > high_4[i]) or \
               (close[i] > ema20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals