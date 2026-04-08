#!/usr/bin/env python3
# 1d_1w_trend_follow_v1
# Hypothesis: Weekly trend-following strategy using EMA21 crossover with volume confirmation on 1d timeframe.
# Captures strong multi-week trends in both bull and bear markets. Uses weekly EMA21 as trend filter and
# daily EMA55 crossover for entry timing with volume spike confirmation. Designed for low trade frequency
# (target: 10-25 trades/year) to minimize fee drag while capturing major trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (EMA21) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA21 on weekly data
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily indicators
    # EMA55 for entry signal
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema55[i]) or np.isnan(avg_volume[i]) or np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema21_1w_aligned[i]
        weekly_downtrend = close[i] < ema21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: weekly trend turns down OR price crosses below EMA55
            if weekly_downtrend or close[i] < ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend turns up OR price crosses above EMA55
            if weekly_uptrend or close[i] > ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Entry conditions with volume confirmation
            if volume_ok:
                # Long entry: price crosses above EMA55 in weekly uptrend
                if weekly_uptrend and close[i] > ema55[i] and close[i-1] <= ema55[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price crosses below EMA55 in weekly downtrend
                elif weekly_downtrend and close[i] < ema55[i] and close[i-1] >= ema55[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals