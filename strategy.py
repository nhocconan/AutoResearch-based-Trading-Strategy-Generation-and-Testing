#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_S1_S4_Breakout_1dTrend_Volume
Hypothesis: Price breaking above S1 or below S4 of daily Camarilla pivots with volume confirmation and daily trend filter works in both bull/bear markets. Target: 15-25 trades/year on 12h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    S1 = prev_close - (range_ * 1.0 / 6)
    S2 = prev_close - (range_ * 2.0 / 6)
    S3 = prev_close - (range_ * 3.0 / 6)
    S4 = prev_close - (range_ * 4.0 / 6)
    R1 = prev_close + (range_ * 1.0 / 6)
    R2 = prev_close + (range_ * 2.0 / 6)
    R3 = prev_close + (range_ * 3.0 / 6)
    R4 = prev_close + (range_ * 4.0 / 6)
    
    # Align pivots to 12h timeframe (only use completed daily bars)
    S1_12h = align_ltf_to_htf(prices, df_1d, S1)
    S4_12h = align_ltf_to_htf(prices, df_1d, S4)
    R1_12h = align_ltf_to_htf(prices, df_1d, R1)
    R4_12h = align_ltf_to_htf(prices, df_1d, R4)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_ltf_to_htf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 24-period average (24 * 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_12h[i]) or np.isnan(S1_12h[i]) or np.isnan(S4_12h[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_12h[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above S1 with volume spike and uptrend
            if vol_spike_val and close[i] > S1_12h[i] and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 with volume spike and downtrend
            elif vol_spike_val and close[i] < S4_12h[i] and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or trend turns down
            if close[i] < S1_12h[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above S4 or trend turns up
            if close[i] > S4_12h[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_S1_S4_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0