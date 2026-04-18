#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1
Hypothesis: Daily breakouts of weekly R1/S1 pivot levels with weekly EMA34 trend filter capture momentum across market cycles. Weekly timeframe provides robust trend filtering to avoid whipsaws in bear markets while allowing participation in strong trends. Designed for low frequency (10-20 trades/year) to minimize fee drag on daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate weekly pivot levels for each week
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    pivot_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * pivot_range) / 12
    s1_level = close_1d - (1.1 * pivot_range) / 12
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: break above R1 with weekly uptrend and volume confirmation
            if price > r1 and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with weekly downtrend and volume confirmation
            elif price < s1 and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below weekly EMA
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above weekly EMA
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0