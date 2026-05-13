#!/usr/bin/env python3
"""
4h_TRIX_Trend_With_Volume_Spike
Hypothesis: TRIX (triple-smoothed EMA) acts as a strong trend filter. A TRIX crossover above/below zero with volume confirmation and aligned 12h trend (close > EMA50) signals continuation. Uses 25% position size to balance risk/return and limit trade frequency (~20-40/year) to minimize fee drag in 4-hour bars.
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
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate TRIX (15,9,9) on close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.pct_change(periods=1) * 100  # Percentage change
    trix_values = trix.fillna(0).values
    
    # 12h trend filter: EMA(50) on close
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average (5 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after TRIX warmup
        if position == 0:
            # LONG: TRIX crosses above zero, volume confirmation, price above 12h EMA50 (uptrend)
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                volume_filter[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, volume confirmation, price below 12h EMA50 (downtrend)
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  volume_filter[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR volume drops
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR volume drops
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals