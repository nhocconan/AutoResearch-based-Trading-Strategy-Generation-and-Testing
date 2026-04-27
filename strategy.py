#!/usr/bin/env python3
"""
6h_WeeklyVWAP_Reversion_12hTrend
Hypothesis: Price tends to revert to weekly VWAP, with 12h trend filter to avoid counter-trend trades. Works in both bull and bear markets as mean reversion occurs in ranging markets while trend filter prevents losses during strong moves. Targets 20-30 trades/year on 6h to minimize fee drag.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for weekly VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly VWAP (5-day period)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price_1d * df_1d['volume']).rolling(window=5, min_periods=5).sum()
    vwap_den = df_1d['volume'].rolling(window=5, min_periods=5).sum()
    weekly_vwap = vwap_num / vwap_den
    
    # Align weekly VWAP to 6h timeframe
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1d, weekly_vwap.values)
    
    # 12h trend filter: EMA20
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_vwap_aligned[i]) or np.isnan(ema20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        vwap = weekly_vwap_aligned[i]
        ema_trend = ema20_12h_aligned[i]
        
        if position == 0:
            # Long: price below VWAP with uptrend
            if close[i] < vwap and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price above VWAP with downtrend
            elif close[i] > vwap and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above VWAP or trend turns down
            if close[i] > vwap or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below VWAP or trend turns up
            if close[i] < vwap or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyVWAP_Reversion_12hTrend"
timeframe = "6h"
leverage = 1.0