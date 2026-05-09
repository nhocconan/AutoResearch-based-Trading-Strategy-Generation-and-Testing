#!/usr/bin/env python3
# Hypothesis: 6h Weekly VWAP Pullback Strategy
# Long when price pulls back to weekly VWAP with bullish daily momentum and volume confirmation
# Short when price rallies to weekly VWAP with bearish daily momentum and volume confirmation
# Uses weekly VWAP as institutional reference, daily EMA50 for trend, and volume spike for conviction
# Designed to work in both trending and ranging markets by exploiting mean reversion to VWAP
# Target: 50-100 total trades over 4 years (12-25/year) with size 0.25

name = "6h_WeeklyVWAP_Pullback_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Typical price and VWAP calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_cum = (typical_price * df_1w['volume']).cumsum()
    vol_cum = df_1w['volume'].cumsum()
    vwap = vwap_cum / vol_cum
    vwap = vwap.shift(1)  # Use previous week's VWAP to avoid look-ahead
    
    # Align weekly VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap.values)
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price pulls back to VWAP from below, daily EMA up, volume spike
            if (close[i] >= vwap_aligned[i] and 
                close[i-1] < vwap_aligned[i-1] and  # Crossed above VWAP
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price rallies to VWAP from above, daily EMA down, volume spike
            elif (close[i] <= vwap_aligned[i] and 
                  close[i-1] > vwap_aligned[i-1] and  # Crossed below VWAP
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves 1% away from VWAP or reverses
            if (close[i] >= vwap_aligned[i] * 1.01) or (close[i] < vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves 1% away from VWAP or reverses
            if (close[i] <= vwap_aligned[i] * 0.99) or (close[i] > vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals