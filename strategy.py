# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI divergence + 1w trend filter + volume spike
# RSI divergence catches reversals at extremes; 1w trend ensures alignment with major trend;
# volume spike confirms institutional participation. Works in both bull and bear markets
# by capturing exhaustion moves and trend continuations with proper filtering.
# Target: 50-150 trades over 4 years (12-37/year) with size 0.25.
name = "6h_RSIDivergence_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    # Price structure for divergence: recent swing high/low
    lookback = 10
    highest_high = pd.Series(high).rolling(window=lookback, center=False).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, center=False).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ema20[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] == lowest_low[i] and 
                       rsi[i] > rsi[i-1] and 
                       rsi[i] < 30 and  # oversold
                       price > ema_1w_aligned[i])  # above weekly trend
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] == highest_high[i] and 
                       rsi[i] < rsi[i-1] and 
                       rsi[i] > 70 and  # overbought
                       price < ema_1w_aligned[i])  # below weekly trend
            
            if bull_div and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            elif bear_div and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 or price breaks below weekly EMA
            if rsi[i] > 50 or price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 or price breaks above weekly EMA
            if rsi[i] < 50 or price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals