#!/usr/bin/env python3
# 12h_Price_Action_Volume_Strategy
# Hypothesis: Price action breakouts with volume confirmation and trend filter.
# Uses 12h chart with 1d trend filter to avoid whipsaw. 
# Long when price breaks above 12h high with volume spike and above 1d EMA200.
# Short when price breaks below 12h low with volume spike and below 1d EMA200.
# Exit when price crosses back through 12h VWAP or trend weakens.
# Designed for 12h timeframe to keep trade count low (target: 50-150 total over 4 years).

name = "12h_Price_Action_Volume_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12h VWAP for exit signal
    # VWAP = cumulative (price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, 0)
    
    # Calculate rolling max/min for breakout levels (20-period)
    lookback = 20
    high_max = np.full_like(high, np.nan)
    low_min = np.full_like(low, np.nan)
    
    for i in range(lookback, n):
        high_max[i] = np.max(high[i-lookback:i])
        low_min[i] = np.min(low[i-lookback:i])
    
    # Calculate volume spike detector (volume > 1.5x 20-period average)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(lookback, n):
        vol_ma[i] = np.mean(volume[i-lookback:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Check for volume spike
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long: price breaks above 12h high, volume spike, above 1d EMA200
            if (close[i] > high_max[i] and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h low, volume spike, below 1d EMA200
            elif (close[i] < low_min[i] and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Long: exit if price crosses below VWAP or breaks below recent low
            if close[i] < vwap[i] or close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above VWAP or breaks above recent high
            if close[i] > vwap[i] or close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals