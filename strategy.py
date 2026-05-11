#!/usr/bin/env python3
"""
1h_4H1D_Trend_With_Volume
Hypothesis: In both bull and bear markets, price tends to follow the higher timeframe trend (4h/1d). 
Enter long when 4h EMA21 > EMA50 and 1d EMA50 > EMA100, with volume confirmation on 1h.
Enter short when 4h EMA21 < EMA50 and 1d EMA50 < EMA100, with volume confirmation.
Use session filter (08-20 UTC) to avoid low-liquidity hours. 
Target 15-30 trades/year (~60-120 total over 4 years) by requiring EMA alignment + volume.
"""

name = "1h_4H1D_Trend_With_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA21 and EMA50 for trend
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA50 and EMA100 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 1h volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Simple stop: exit if trend breaks
                if position == 1:
                    if ema21_4h_aligned[i] <= ema50_4h_aligned[i] or ema50_1d_aligned[i] <= ema100_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.20
                else:  # position == -1
                    if ema21_4h_aligned[i] >= ema50_4h_aligned[i] or ema50_1d_aligned[i] >= ema100_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.20
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        # Trend conditions
        bullish_4h = ema21_4h_aligned[i] > ema50_4h_aligned[i]
        bearish_4h = ema21_4h_aligned[i] < ema50_4h_aligned[i]
        bullish_1d = ema50_1d_aligned[i] > ema100_1d_aligned[i]
        bearish_1d = ema50_1d_aligned[i] < ema100_1d_aligned[i]
        
        if position == 0:
            # Look for new entries
            if in_session[i] and vol_confirm:
                # Long: both 4h and 1d bullish
                if bullish_4h and bullish_1d:
                    signals[i] = 0.20
                    position = 1
                    entry_price = prices['close'].iloc[i]
                # Short: both 4h and 1d bearish
                elif bearish_4h and bearish_1d:
                    signals[i] = -0.20
                    position = -1
                    entry_price = prices['close'].iloc[i]
        else:
            # Manage existing position
            if position == 1:
                # Exit long if either timeframe turns bearish
                if not (bullish_4h and bullish_1d):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short if either timeframe turns bullish
                if not (bearish_4h and bearish_1d):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals