#!/usr/bin/env python3
"""
1d_1w_Camarilla_Reverse_Trend
Hypothesis: Camarilla pivot levels on daily chart provide reversal signals during weekly trend extremes.
In strong weekly trends (price > weekly EMA50), look for reversals at Camarilla H3/L3 levels.
In weak weekly trends (price < weekly EMA50), look for continuations at H4/L4 levels.
Uses volume confirmation to filter false signals. Works in both bull and bear markets by adapting to weekly trend context.
Target: 15-25 trades/year on 1d (60-100 total over 4 years).
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
    
    # Get daily data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H3 and L3 are primary reversal levels
    # H4 and L4 are breakout/continuation levels
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.125 * (prev_high - prev_low)
    L3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to daily timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below weekly EMA50
        weekly_uptrend = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False
        # Simplified: use aligned weekly EMA for current day
        weekly_uptrend_today = close[i] > ema_50_1w_aligned[i]
        
        # Long conditions
        long_signal = False
        if weekly_uptrend_today:
            # In uptrend: look for reversals at H3 (sell the bounce)
            if close[i] <= H3_aligned[i] and volume_expansion[i]:
                long_signal = False  # Actually a short signal in uptrend at H3
            # In uptrend: look for continuations above H4 (breakout)
            elif close[i] > H4_aligned[i] and volume_expansion[i]:
                long_signal = True
        else:
            # In downtrend/sideways: look for reversals at L3 (buy the dip)
            if close[i] >= L3_aligned[i] and volume_expansion[i]:
                long_signal = True
            # In downtrend: look for continuations below L4 (breakdown)
            elif close[i] < L4_aligned[i] and volume_expansion[i]:
                long_signal = False  # Actually a short signal
        
        # Short conditions
        short_signal = False
        if weekly_uptrend_today:
            # In uptrend: look for reversals at H3 (sell the bounce)
            if close[i] <= H3_aligned[i] and volume_expansion[i]:
                short_signal = True
            # In uptrend: look for continuations above H4 (breakout)
            elif close[i] > H4_aligned[i] and volume_expansion[i]:
                short_signal = False  # Actually a long signal
        else:
            # In downtrend/sideways: look for reversals at L3 (buy the dip)
            if close[i] >= L3_aligned[i] and volume_expansion[i]:
                short_signal = False  # Actually a long signal
            # In downtrend: look for continuations below L4 (breakdown)
            elif close[i] < L4_aligned[i] and volume_expansion[i]:
                short_signal = True
        
        # Special case: ranging market (price near weekly EMA) - mean reversion at H3/L3
        if abs(close[i] - ema_50_1w_aligned[i]) < (ema_50_1w_aligned[i] * 0.02):  # Within 2% of weekly EMA
            # Mean reversion: sell at H3, buy at L3
            if close[i] > H3_aligned[i] and volume_expansion[i]:
                short_signal = True
                long_signal = False
            elif close[i] < L3_aligned[i] and volume_expansion[i]:
                long_signal = True
                short_signal = False
        
        # Update position
        if long_signal and not short_signal:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif short_signal and not long_signal:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Reverse_Trend"
timeframe = "1d"
leverage = 1.0