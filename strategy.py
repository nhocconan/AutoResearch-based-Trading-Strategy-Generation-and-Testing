#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels with volume confirmation and weekly trend filter.
# Long when price breaks above H3 with volume > 1.5x average and weekly trend up.
# Short when price breaks below L3 with volume > 1.5x average and weekly trend down.
# Exit when price returns to pivot level or volume drops.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_v1"
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
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend direction
    close_1w_series = pd.Series(close_1w)
    ema21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 21) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(ema21_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for today
        # Using yesterday's OHLC
        if i == 0:
            continue  # Skip first bar
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_val = phigh - plow
        
        # Camarilla levels
        h3 = pivot + (range_val * 1.1 / 2)
        l3 = pivot - (range_val * 1.1 / 2)
        
        if position == 1:  # Long position
            # Exit: Price below H3 or volume drops
            if close[i] < h3 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above L3 or volume drops
            if close[i] > l3 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above H3 with volume surge and weekly uptrend
            if (close[i] > h3 and 
                vol_surge[i] and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 with volume surge and weekly downtrend
            elif (close[i] < l3 and 
                  vol_surge[i] and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals