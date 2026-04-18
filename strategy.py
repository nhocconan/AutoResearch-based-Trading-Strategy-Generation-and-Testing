#!/usr/bin/env python3
"""
1d_1w_Woodies_CCI_Divergence
Hypothesis: Combines Woodies CCI with divergence detection on 1d timeframe and uses 1w trend as filter.
Trades CCI extremes with divergence in direction of weekly trend. Designed to work in both bull and bear
markets by filtering with 1w trend. Target: 10-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Woodies CCI (20-period)
    typical_price = (high + low + close) / 3.0
    tp_mean = np.full(n, np.nan)
    tp_dev = np.full(n, np.nan)
    cci = np.full(n, np.nan)
    
    for i in range(20, n):
        tp_slice = typical_price[i-20:i+1]
        tp_mean[i] = np.mean(tp_slice)
        tp_dev[i] = np.mean(np.abs(tp_slice - tp_mean[i]))
        if tp_dev[i] > 0:
            cci[i] = (typical_price[i] - tp_mean[i]) / (0.015 * tp_dev[i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2/35) + (ema_34_1w[i-1] * 33/35)
    
    # Align EMA34 to 1d timeframe (wait for bar close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Detect bullish/bearish divergence
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for divergence over last 10 bars
    lookback = 10
    for i in range(lookback, n):
        if np.isnan(cci[i]) or np.isnan(cci[i-lookback]):
            continue
            
        # Bullish divergence: price makes lower low, CCI makes higher low
        if (low[i] < low[i-lookback] and 
            cci[i] > cci[i-lookback] and
            cci[i] < -100):  # Only in oversold territory
            # Find local lows
            price_low_idx = i - np.argmin(low[i-lookback:i+1])
            cci_low_idx = i - np.argmin(cci[i-lookback:i+1])
            if price_low_idx == cci_low_idx:  # Same bar
                bullish_div[i] = True
                
        # Bearish divergence: price makes higher high, CCI makes lower high
        if (high[i] > high[i-lookback] and 
            cci[i] < cci[i-lookback] and
            cci[i] > 100):  # Only in overbought territory
            # Find local highs
            price_high_idx = i - np.argmax(high[i-lookback:i+1])
            cci_high_idx = i - np.argmax(cci[i-lookback:i+1])
            if price_high_idx == cci_high_idx:  # Same bar
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: CCI < -200 with bullish divergence and price above weekly EMA34
            if (cci[i] < -200 and bullish_div[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: CCI > 200 with bearish divergence and price below weekly EMA34
            elif (cci[i] > 200 and bearish_div[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: CCI crosses above zero or drops below -300 (extreme oversold)
            if cci[i] > 0 or cci[i] < -300:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses below zero or rises above 300 (extreme overbought)
            if cci[i] < 0 or cci[i] > 300:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Woodies_CCI_Divergence"
timeframe = "1d"
leverage = 1.0