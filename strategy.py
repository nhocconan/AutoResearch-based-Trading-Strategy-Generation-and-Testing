#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_With_Volume_Confirmation_v2
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise,
providing reliable trend direction in both trending and ranging markets.
Combined with weekly trend filter (KAMA slope) and daily volume confirmation,
this strategy captures sustained moves while avoiding false signals in chop.
Designed for 1d timeframe with weekly trend filter to reduce whipsaw in bear markets.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=1))
        change = np.insert(change, 0, 0)  # align length
        
        volatility = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                volatility[i] = 0
            else:
                volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period+1):i+1])))
        
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_daily = kama(close, er_period=10, fast=2, slow=30)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    kama_weekly = kama(close_1w, er_period=10, fast=2, slow=30)
    kama_weekly_aligned = align_htf_to_ltf(prices, df_1w, kama_weekly)
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_ma_20 = np.zeros(n)
    vol_series = pd.Series(volume)
    for i in range(n):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = vol_series.iloc[i-19:i+1].mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_daily[i]) or np.isnan(kama_weekly_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above daily KAMA AND weekly KAMA rising AND volume expansion
        long_condition = (close[i] > kama_daily[i] and 
                         kama_weekly_aligned[i] > kama_weekly_aligned[i-1] and
                         volume_expansion[i])
        
        # Short condition: price below daily KAMA AND weekly KAMA falling AND volume expansion
        short_condition = (close[i] < kama_daily[i] and 
                          kama_weekly_aligned[i] < kama_weekly_aligned[i-1] and
                          volume_expansion[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_Trend_Filter_With_Volume_Confirmation_v2"
timeframe = "1d"
leverage = 1.0