#!/usr/bin/env python3
# 1d_ema200_trend_weekly_ema_filter_v1
# Hypothesis: Price above/below daily EMA200 with weekly EMA200 confirmation captures
# primary trend direction. Only trade when weekly trend agrees with daily to avoid
# counter-trend whipsaws. Volume > 1.5x average confirms momentum. Target: 10-20 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema200_trend_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200
    daily_ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly EMA200 (HTF)
    df_weekly = get_htf_data(prices, '1w')
    weekly_ema200_raw = pd.Series(df_weekly['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200 = align_htf_to_ltf(prices, df_weekly, weekly_ema200_raw)
    
    # Volume average (20-day)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(daily_ema200[i]) or np.isnan(weekly_ema200[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below daily EMA200 OR weekly trend turns bearish
            if close[i] < daily_ema200[i] or weekly_ema200[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above daily EMA200 OR weekly trend turns bullish
            if close[i] > daily_ema200[i] or weekly_ema200[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: Price above both EMAs with volume
            if close[i] > daily_ema200[i] and close[i] > weekly_ema200[i] and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below both EMAs with volume
            elif close[i] < daily_ema200[i] and close[i] < weekly_ema200[i] and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals