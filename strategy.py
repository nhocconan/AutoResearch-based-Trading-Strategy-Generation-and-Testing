#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Confluence_Breakout
Hypothesis: Weekly pivot points act as strong support/resistance. On 6h timeframe, we look for breakouts above weekly R1 or below weekly S1 with volume confirmation and 1d trend filter (EMA34). Weekly timeframe provides structure that works in both bull and bear markets, while 6h allows precise entry timing. Discrete position sizing (0.25) limits fee drag. Target: 12-37 trades/year (~50-150 over 4 years).
"""

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
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for pivot points
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Standard weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_p = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = 2 * weekly_p - l_1w
    weekly_s1 = 2 * weekly_p - h_1w
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (strict for low trade frequency)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need weekly data (1w pivot calculation) + 1d EMA34 (34) + volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Weekly R1/S1 breakout with 1d EMA34 trend filter and volume spike
            # Long: price closes above weekly R1 AND above 1d EMA34 (uptrend) AND volume spike
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below weekly S1 AND below 1d EMA34 (downtrend) AND volume spike
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions:
            # 1. Price touches weekly S1 (opposite pivot level)
            # 2. 1d EMA34 turns bearish (price below EMA)
            exit_condition = (close_val < s1_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches weekly R1 (opposite pivot level)
            # 2. 1d EMA34 turns bullish (price above EMA)
            exit_condition = (close_val > r1_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Confluence_Breakout"
timeframe = "6h"
leverage = 1.0