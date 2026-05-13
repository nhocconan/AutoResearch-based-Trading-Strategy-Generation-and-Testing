#!/usr/bin/env python3
"""
1d_Weekly_Trix_Trend_With_Volume_Spike
Hypothesis: TRIX on weekly timeframe acts as a strong trend filter for daily signals. A weekly TRIX crossover above/below zero with daily volume confirmation and daily price above/below weekly EMA50 signals continuation. Uses 0.30 position size to balance risk/return and limit trade frequency (~10-20/year) to minimize fee drag in daily bars.
"""

name = "1d_Weekly_Trix_Trend_With_Volume_Spike"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate TRIX (15,9,9) on weekly close
    ema1 = pd.Series(df_weekly['close']).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.pct_change(periods=1) * 100  # Percentage change
    trix_values = trix.fillna(0).values
    
    # Weekly trend filter: EMA(50) on weekly close
    ema50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume confirmation: current daily volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after TRIX warmup
        if position == 0:
            # LONG: Weekly TRIX crosses above zero, volume confirmation, daily price above weekly EMA50 (uptrend)
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                volume_filter[i] and 
                close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Weekly TRIX crosses below zero, volume confirmation, daily price below weekly EMA50 (downtrend)
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  volume_filter[i] and 
                  close[i] < ema50_weekly_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly TRIX crosses below zero OR volume drops
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Weekly TRIX crosses above zero OR volume drops
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals