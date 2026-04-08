#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_volume
Hypothesis: Camarilla pivot levels on daily chart provide strong support/resistance.
Breakouts above/below H4/L4 levels with volume confirmation and 1w trend filter.
Works in both bull/bear markets: breakouts capture momentum, trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume"
timeframe = "12h"
leverage = 1.0

def camarilla_pivot(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns: H4, H3, H2, H1, L1, L2, L3, L4
    """
    typical = (high + low + close) / 3
    range_val = high - low
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    H2 = close + range_val * 1.1 / 6
    H1 = close + range_val * 1.1 / 12
    L1 = close - range_val * 1.1 / 12
    L2 = close - range_val * 1.1 / 6
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    return H4, H3, H2, H1, L1, L2, L3, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    H4 = np.full_like(close_1d, np.nan)
    H3 = np.full_like(close_1d, np.nan)
    H2 = np.full_like(close_1d, np.nan)
    H1 = np.full_like(close_1d, np.nan)
    L1 = np.full_like(close_1d, np.nan)
    L2 = np.full_like(close_1d, np.nan)
    L3 = np.full_like(close_1d, np.nan)
    L4 = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day's data
            H4[i], H3[i], H2[i], H1[i], L1[i], L2[i], L3[i], L4[i] = camarilla_pivot(
                high_1d[i-1], low_1d[i-1], close_1d[i-1]
            )
    
    # Forward fill the levels (they remain valid until next day's calculation)
    H4_series = pd.Series(H4)
    H3_series = pd.Series(H3)
    H2_series = pd.Series(H2)
    H1_series = pd.Series(H1)
    L1_series = pd.Series(L1)
    L2_series = pd.Series(L2)
    L3_series = pd.Series(L3)
    L4_series = pd.Series(L4)
    
    H4_ffilled = H4_series.ffill().values
    H3_ffilled = H3_series.ffill().values
    H2_ffilled = H2_series.ffill().values
    H1_ffilled = H1_series.ffill().values
    L1_ffilled = L1_series.ffill().values
    L2_ffilled = L2_series.ffill().values
    L3_ffilled = L3_series.ffill().values
    L4_ffilled = L4_series.ffill().values
    
    # Align 1d Camarilla levels to 12h
    H4_12h = align_htf_to_ltf(prices, df_1d, H4_ffilled)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3_ffilled)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2_ffilled)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1_ffilled)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1_ffilled)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2_ffilled)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3_ffilled)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4_ffilled)
    
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
    
    # Align 1w trend to 12h
    trend_1w_up_12h = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_12h = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or
            np.isnan(trend_1w_up_12h[i]) or np.isnan(trend_1w_down_12h[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below H3 OR 1w trend turns down
            if (close[i] < H3_12h[i]) or trend_1w_down_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price crosses above L3 OR 1w trend turns up
            if (close[i] > L3_12h[i]) or trend_1w_up_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above H4 with volume + 1w uptrend
            if (close[i] > H4_12h[i]) and volume_filter[i] and trend_1w_up_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume + 1w downtrend
            elif (close[i] < L4_12h[i]) and volume_filter[i] and trend_1w_down_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals