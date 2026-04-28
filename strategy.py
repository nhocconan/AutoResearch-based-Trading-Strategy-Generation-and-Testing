#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2
Hypothesis: Tightening the previous version by requiring stronger volume confirmation (>3x 20-period average) and adding a 4-hour EMA50 trend filter to reduce false breakouts. Targets 25-35 trades/year with improved BTC/ETH performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h EMA50 for additional trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    R1 = typical_price + (range_ * 1.1 / 12)
    S1 = typical_price - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # Volume confirmation: >3.0x 20-period MA (stricter than before)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_4h[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: price above/both EMAs for long, below/both for short
        uptrend = close[i] > ema_34_1d_aligned[i] and close[i] > ema_50_4h[i]
        downtrend = close[i] < ema_34_1d_aligned[i] and close[i] < ema_50_4h[i]
        
        # Volume confirmation (>3x average)
        vol_confirm = volume[i] > (3.0 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = close[i] > R1_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S1_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint
        midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0