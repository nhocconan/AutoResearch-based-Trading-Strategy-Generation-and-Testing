#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla H3/L3 levels with 1d EMA200 filter
    # 4h Camarilla provides institutional support/resistance levels
    # 1d EMA200 defines primary trend (bull/bear regime)
    # Session filter (08-20 UTC) reduces noise outside active trading hours
    # Target: 15-35 trades/year (60-140 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous 4h bar's Camarilla levels
    # H3/L3 are key levels for reversal/continuation
    camarilla_h3 = np.full(len(high_4h), np.nan)
    camarilla_l3 = np.full(len(low_4h), np.nan)
    
    for i in range(1, len(high_4h)):
        # Previous 4h bar's range
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        rang = ph - pl
        
        camarilla_h3[i] = pc + (rang * 1.1/4)  # H3 = Close + 1.1*Range/4
        camarilla_l3[i] = pc - (rang * 1.1/4)  # L3 = Close - 1.1*Range/4
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA200 on daily timeframe
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    # Align HTF indicators to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on 1d EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Entry logic: Camarilla H3/L3 breakouts with trend filter
        long_entry = (close[i] > h3_aligned[i]) and uptrend
        short_entry = (close[i] < l3_aligned[i]) and downtrend
        
        # Exit logic: return to opposite Camarilla level or trend reversal
        long_exit = (close[i] < l3_aligned[i]) or (not uptrend)
        short_exit = (close[i] > h3_aligned[i]) or (not downtrend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_h3l3_ema200_session_v1"
timeframe = "1h"
leverage = 1.0