#!/usr/bin/env python3
"""
4h_TRIX_Trend_With_Volume_Spike
Hypothesis: TRIX (Triple Exponential Moving Average) captures momentum with less noise.
In bull markets, TRIX > 0 with volume spikes indicates strength. In bear markets, TRIX < 0 with volume spikes indicates weakness.
Uses 1-day trend filter and volume confirmation to reduce false signals.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
Works in both bull and bear regimes by following momentum with institutional volume backing.
"""

name = "4h_TRIX_Trend_With_Volume_Spike"
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
    
    # Get daily data for 1-day trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate TRIX on 4h close: TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100
    trix = trix.fillna(0).values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: TRIX positive (bullish momentum) with volume spike and above 1-day EMA34
            if (trix[i] > 0.0 and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX negative (bearish momentum) with volume spike and below 1-day EMA34
            elif (trix[i] < 0.0 and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative or price drops below 1-day EMA34
            if (trix[i] < 0.0 or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive or price rises above 1-day EMA34
            if (trix[i] > 0.0 or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals