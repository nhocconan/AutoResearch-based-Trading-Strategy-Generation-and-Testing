#/usr/bin/env python3
"""
1d_Weekly_Channel_Breakout_1wTrend
Hypothesis: Price breaks the weekly high/low calculated from 1w data, with 1w EMA20 trend filter and volume confirmation.
Breakouts from weekly extremes capture sustained momentum across market cycles, while the weekly trend filter ensures alignment
with the longer-term direction. Volume confirmation filters false breakouts. Works in bull/bear by trading only in the
direction of the weekly trend. Target: 10-20 trades/year (40-80 total) to minimize fee drag.
"""

name = "1d_Weekly_Channel_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly extremes and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly high/low lookback (5 periods = 5 weeks)
    lookback = 5
    high_weekly = np.full(len(high_1w), np.nan)
    low_weekly = np.full(len(low_1w), np.nan)
    
    if len(high_1w) >= lookback:
        for i in range(lookback, len(high_1w)):
            high_weekly[i] = np.max(high_1w[i-lookback:i])
            low_weekly[i] = np.min(low_1w[i-lookback:i])
    
    # Weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Weekly volume SMA5 for volume confirmation
    vol_sma5_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 5:
        vol_sma5_1w[4] = np.mean(volume_1w[:5])
        for i in range(5, len(volume_1w)):
            vol_sma5_1w[i] = (vol_sma5_1w[i-1] * 4 + volume_1w[i]) / 5
    
    # Align 1w indicators to 1d
    high_weekly_aligned = align_htf_to_ltf(prices, df_1w, high_weekly)
    low_weekly_aligned = align_htf_to_ltf(prices, df_1w, low_weekly)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    vol_sma5_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma5_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(high_weekly_aligned[i]) or np.isnan(low_weekly_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_sma5_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1w volume (scaled)
        # 1w = 5 x 1d bars, so scale weekly volume to daily equivalent
        vol_1w_scaled = vol_sma5_1w_aligned[i] / 5.0  # Average 1d-equivalent volume from 1w data
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to weekly levels
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        price_above_weekly_high = close[i] > high_weekly_aligned[i]
        price_below_weekly_low = close[i] < low_weekly_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly high, in uptrend, with volume
            if price_above_weekly_high and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low, in downtrend, with volume
            elif price_below_weekly_low and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly high or trend turns down
            if not price_above_weekly_high or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly low or trend turns up
            if not price_below_weekly_low or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals