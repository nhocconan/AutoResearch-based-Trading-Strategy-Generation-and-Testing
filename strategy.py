#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_trend_v1
# Hypothesis: Daily Donchian breakout with weekly EMA trend filter and volume confirmation.
# Long when price breaks above 20-day high with volume > 1.5x 20-day average and price above weekly EMA20.
# Short when price breaks below 20-day low with volume > 1.5x 20-day average and price below weekly EMA20.
# Designed for 10-25 trades/year to minimize fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(period-1, n):
        highest_high[i] = np.max(high[i-(period-1):i+1])
        lowest_low[i] = np.min(low[i-(period-1):i+1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(period-1, n):
        vol_avg[i] = np.mean(volume[i-(period-1):i+1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = period-1  # Start when Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or weekly trend turns down
            if close[i] < lowest_low[i] or close[i] < ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or weekly trend turns up
            if close[i] > highest_high[i] or close[i] > ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_surge = volume[i] > 1.5 * vol_avg[i]
            
            # Long entry: break above 20-day high with volume surge and weekly uptrend
            if (close[i] > highest_high[i] and 
                vol_surge and 
                close[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: break below 20-day low with volume surge and weekly downtrend
            elif (close[i] < lowest_low[i] and 
                  vol_surge and 
                  close[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals