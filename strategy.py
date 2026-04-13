#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Pivot_Momentum
Hypothesis: Combines Camarilla pivot levels from daily with weekly trend filter and volume confirmation.
In bull markets (price > weekly EMA50), buy near L3 support with volume spike.
In bear markets (price < weekly EMA50), sell short near H3 resistance with volume spike.
Uses 4h timeframe for entries to avoid overtrading. Target: 20-40 trades/year.
Works in both bull and bear by adapting to weekly trend direction.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low)
    # H2 = close + 0.5*(high-low), L2 = close - 0.5*(high-low)
    # H1 = close + 0.25*(high-low), L1 = close - 0.25*(high-low)
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.5 * hl_range
    L4 = close_1d - 1.5 * hl_range
    H3 = close_1d + 1.0 * hl_range
    L3 = close_1d - 1.0 * hl_range
    H2 = close_1d + 0.5 * hl_range
    L2 = close_1d - 0.5 * hl_range
    H1 = close_1d + 0.25 * hl_range
    L1 = close_1d - 0.25 * hl_range
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    # Trend: bull if close > EMA50, bear if close < EMA50
    trend_bull = close_1w > ema_50_1w
    
    # Get 4h data for entry timing and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_spike = volume_4h > (vol_ma_20_4h * 2.0)  # Volume spike > 2x average
    
    # Align all signals to 4h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    trend_bull_aligned = align_htf_to_ltf(prices, df_1w, trend_bull)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or \
           np.isnan(trend_bull_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: bull trend + price near L3 support + volume spike
        if trend_bull_aligned[i]:
            # Enter long when price touches or goes below L3 with volume spike
            if low[i] <= L3_aligned[i] and volume_spike_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Exit long when price reaches H2 (profit target) or reverses below L2
            elif position == 1 and (high[i] >= H2_aligned[i] or low[i] < L2_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Hold long
            elif position == 1:
                signals[i] = position_size
            else:
                signals[i] = 0.0
        
        # Short conditions: bear trend + price near H3 resistance + volume spike
        elif not trend_bull_aligned[i]:
            # Enter short when price touches or goes above H3 with volume spike
            if high[i] >= H3_aligned[i] and volume_spike_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit short when price reaches L2 (profit target) or reverses above H2
            elif position == -1 and (low[i] <= L2_aligned[i] or high[i] > H2_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Hold short
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        
        # No clear trend - stay flat
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Camarilla_Pivot_Momentum"
timeframe = "4h"
leverage = 1.0