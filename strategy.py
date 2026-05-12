#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume_Squeeze"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data once for daily Bollinger Bands (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_pct = bb_width / bb_middle
    
    # Calculate daily Bollinger Band width percentile (252-day lookback)
    bb_width_series = pd.Series(bb_width_pct)
    bb_width_percentile = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Load 1d data once for daily Camarilla pivots (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r1 = p + (high_1d - low_1d) * 1.1 / 12
    s1 = p - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(bb_width_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 1w EMA50 + volume spike + low volatility (squeeze)
            if (close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i] and bb_width_percentile_aligned[i] < 0.3):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below 1w EMA50 + volume spike + low volatility (squeeze)
            elif (close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i] and bb_width_percentile_aligned[i] < 0.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or volatility expands
            if close[i] < s1_aligned[i] or bb_width_percentile_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 or volatility expands
            if close[i] > r1_aligned[i] or bb_width_percentile_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals