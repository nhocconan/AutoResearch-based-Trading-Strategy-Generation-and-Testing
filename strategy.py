#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and trend alignment (price > EMA50 for longs, < EMA50 for shorts) capture momentum with controlled risk. Uses 30% position size to balance return and drawdown. Designed for low trade frequency (~25-40/year) to minimize fee drag.
"""

name = "4h_Donchian_Breakout_Volume_Trend"
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
    
    # Get 4h data for EMA50 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Donchian channels (20-period) - calculated on price data
    # Highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian high, volume confirmation, price above EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low, volume confirmation, price below EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian high (failed breakout) OR trend turns bearish
            if (close[i] < donchian_high[i] or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian low (failed breakdown) OR trend turns bullish
            if (close[i] > donchian_low[i] or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals