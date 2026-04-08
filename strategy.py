#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_volume_v1
Hypothesis: Camarilla pivot levels on 1d with volume confirmation and weekly trend filter.
- Entry: Price touches Camarilla support/resistance on 1d + volume spike + 1w trend alignment
- Exit: Opposite Camarilla level touch or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed to work in ranging markets (Camarilla reversals) and trending markets (breakouts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    L4 = close_val + (1.1/12) * range_val
    L3 = close_val + (1.1/6) * range_val
    L2 = close_val + (1.1/4) * range_val
    L1 = close_val + (1.1/12) * range_val
    H1 = close_val - (1.1/12) * range_val
    H2 = close_val - (1.1/4) * range_val
    H3 = close_val - (1.1/6) * range_val
    H4 = close_val - (1.1/12) * range_val
    return L4, L3, L2, L1, H1, H2, H3, H4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    L4_1d = np.full(len(close_1d), np.nan)
    L3_1d = np.full(len(close_1d), np.nan)
    L2_1d = np.full(len(close_1d), np.nan)
    L1_1d = np.full(len(close_1d), np.nan)
    H1_1d = np.full(len(close_1d), np.nan)
    H2_1d = np.full(len(close_1d), np.nan)
    H3_1d = np.full(len(close_1d), np.nan)
    H4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        L4, L3, L2, L1, H1, H2, H3, H4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        L4_1d[i] = L4
        L3_1d[i] = L3
        L2_1d[i] = L2
        L1_1d[i] = L1
        H1_1d[i] = H1
        H2_1d[i] = H2
        H3_1d[i] = H3
        H4_1d[i] = H4
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1d Camarilla and 1w trend to 12h
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    L2_1d_aligned = align_htf_to_ltf(prices, df_1d, L2_1d)
    L1_1d_aligned = align_htf_to_ltf(prices, df_1d, L1_1d)
    H1_1d_aligned = align_htf_to_ltf(prices, df_1d, H1_1d)
    H2_1d_aligned = align_htf_to_ltf(prices, df_1d, H2_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(L4_1d_aligned[i]) or np.isnan(H4_1d_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches H3/H4 (resistance) OR 1w trend turns down
            if (close[i] >= H3_1d_aligned[i] or close[i] >= H4_1d_aligned[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches L3/L4 (support) OR 1w trend turns up
            if (close[i] <= L3_1d_aligned[i] or close[i] <= L4_1d_aligned[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price touches L1/L2 (support) + volume + 1w uptrend
            if ((close[i] <= L1_1d_aligned[i] or close[i] <= L2_1d_aligned[i]) and
                volume_filter[i] and trend_1w_up_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H1/H2 (resistance) + volume + 1w downtrend
            elif ((close[i] >= H1_1d_aligned[i] or close[i] >= H2_1d_aligned[i]) and
                  volume_filter[i] and trend_1w_down_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals