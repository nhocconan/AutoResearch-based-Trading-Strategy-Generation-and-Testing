#!/usr/bin/env python3
"""
4h_Acceleration_Bounce
Hypothesis: In volatile crypto markets, sharp momentum reversals after extreme moves often precede mean-reverting bounces. 
This strategy identifies overextended moves using 4-hour RSI divergence from price extremes, confirmed by volume spikes 
and aligned with the 1-day trend (price > EMA50 for longs, < EMA50 for shorts). 
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets by combining 
contrarian entry with trend filter to avoid counter-trend traps. 
Low frequency (~20-40 trades/year) minimizes fee drag.
"""

name = "4h_Acceleration_Bounce"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) - momentum oscillator
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral fill for warmup
    
    # 1-day trend filter: EMA(50) on close
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Price extremes: recent 10-bar high/low for overextension check
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: RSI oversold (<30) + price near recent low + volume spike + uptrend (price > EMA50)
            if (rsi_values[i] < 30 and 
                close[i] <= lowest_10[i] * 1.005 and  # within 0.5% of recent low
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) + price near recent high + volume spike + downtrend (price < EMA50)
            elif (rsi_values[i] > 70 and 
                  close[i] >= highest_10[i] * 0.995 and  # within 0.5% of recent high
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) OR trend breaks (price < EMA50)
            if rsi_values[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) OR trend breaks (price > EMA50)
            if rsi_values[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals