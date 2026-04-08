#!/usr/bin/env python3
# daily_price_action_v1
# Hypothesis: Daily chart strategy using price action at weekly 200 EMA with volume confirmation.
# Long when price crosses above weekly 200 EMA with volume > 1.5x average.
# Short when price crosses below weekly 200 EMA with volume > 1.5x average.
# Exit when price crosses back across weekly 200 EMA.
# Target: 15-25 trades/year, works in both bull and bear markets by following weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_price_action_v1"
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
    
    # Weekly 200 EMA for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(200, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly 200 EMA
            if close[i] < ema200_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly 200 EMA
            if close[i] > ema200_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above weekly 200 EMA with volume surge
            if (close[i] > ema200_weekly_aligned[i] and 
                close[i-1] <= ema200_weekly_aligned[i-1] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below weekly 200 EMA with volume surge
            elif (close[i] < ema200_weekly_aligned[i] and 
                  close[i-1] >= ema200_weekly_aligned[i-1] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals