#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1-day Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance levels. 
# Buy near L3/L4 with bullish weekly trend and volume expansion. 
# Sell near H3/H4 with bearish weekly trend and volume expansion.
# Uses tight entry conditions (pivot touches + volume + trend) to limit trades (~15-25/year) 
# and avoid fee drag. Works in bull markets by buying dips in uptrend and in bear markets 
# by selling rallies in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly close for trend filter (using simple price position vs 50-period SMA)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after minimum lookback for pivot calculation
        # Need at least 20 days of data to calculate meaningful pivots
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla pivot levels for previous day
        # Using high, low, close from previous day (i-1)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Calculate pivot point
        pivot = (phigh + plow + pclose) / 3.0
        
        # Calculate Camarilla levels
        range_val = phigh - plow
        if range_val <= 0:
            # Skip if no range (unlikely but possible)
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla levels
        h4 = pclose + (range_val * 1.1 / 2)
        h3 = pclose + (range_val * 1.1 / 4)
        h2 = pclose + (range_val * 1.1 / 6)
        h1 = pclose + (range_val * 1.1 / 12)
        l1 = pclose - (range_val * 1.1 / 12)
        l2 = pclose - (range_val * 1.1 / 6)
        l3 = pclose - (range_val * 1.1 / 4)
        l4 = pclose - (range_val * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_avg_20 = np.mean(volume[i-20:i]) if i >= 20 else volume[i]
            vol_confirm = volume[i] > (1.5 * vol_avg_20)
        else:
            vol_confirm = False
        
        # Trend filter: price above/below weekly SMA50
        uptrend = close[i] > sma_50_1w_aligned[i]
        downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Entry logic: Near pivot levels with volume and trend alignment
        # Long near L3/L4 in uptrend
        if uptrend and vol_confirm:
            if (low[i] <= l3 and close[i] > l1) or (low[i] <= l4 and close[i] > l1):
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short near H3/H4 in downtrend
            elif (high[i] >= h3 and close[i] < h1) or (high[i] >= h4 and close[i] < h1):
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        # In downtrend, look for shorts near resistance
        elif downtrend and vol_confirm:
            if (high[i] >= h3 and close[i] < h1) or (high[i] >= h4 and close[i] < h1):
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            # Look for longs near support in downtrend (counter-trend bounces)
            elif (low[i] <= l3 and close[i] > l1) or (low[i] <= l4 and close[i] > l1):
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or choppy market - hold or flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals