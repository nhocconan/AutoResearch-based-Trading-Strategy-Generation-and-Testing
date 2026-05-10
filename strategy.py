# #!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend
Hypothesis: Price breaks weekly pivot-based resistance (R1) or support (S1) levels, with weekly EMA50 trend filter and daily volume confirmation. 
Weekly pivots act as significant support/resistance; breakouts with volume and trend alignment capture directional moves.
Works in bull/bear by filtering trades in direction of weekly trend.
Target: 10-25 trades/year (40-100 total) to minimize fee drag.
"""

name = "1d_WeeklyPivot_Breakout_1wTrend"
timeframe = "1d"
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
    
    # 1w data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly pivot-based levels from prior week: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    weekly_r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    weekly_s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    # Weekly EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Daily volume SMA20 for volume confirmation
    vol_sma20_d = np.full(len(prices), np.nan)
    if len(prices) >= 20:
        vol_sma20_d[19] = np.mean(volume[:20])
        for i in range(20, len(prices)):
            vol_sma20_d[i] = (vol_sma20_d[i-1] * 19 + volume[i]) / 20
    
    # Align weekly indicators to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average daily volume
        volume_confirm = volume[i] > 1.5 * vol_sma20_d[i]
        
        # Trend and price relative to weekly pivot levels
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals