#!/usr/bin/env python3
"""
4h_1d_ema_aligned_breakout
Strategy: 4h EMA50/200 alignment with price breaking above/below 1d EMA20 for entry
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h EMA50 above EMA200 as uptrend filter, EMA50 below EMA200 as downtrend filter. Enters long when price crosses above 1d EMA20 in uptrend, short when price crosses below 1d EMA20 in downtrend. Exits when price crosses back below/above 1d EMA20. Uses volume confirmation (1.5x average volume) to filter false breakouts. Designed to capture trend continuation moves with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_aligned_breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 and EMA200 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d EMA20 for entry/exit
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        uptrend = ema_50[i] > ema_200[i]
        downtrend = ema_50[i] < ema_200[i]
        
        # Entry conditions
        long_signal = (price_close > ema_20_1d_aligned[i]) and uptrend and vol_confirm[i]
        short_signal = (price_close < ema_20_1d_aligned[i]) and downtrend and vol_confirm[i]
        
        # Exit conditions: price crosses back below/above 1d EMA20
        exit_long = position == 1 and (price_close < ema_20_1d_aligned[i])
        exit_short = position == -1 and (price_close > ema_20_1d_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals