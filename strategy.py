#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day volume-weighted average price (VWAP) with 1-week trend filter.
# VWAP acts as dynamic support/resistance, with price tending to revert to VWAP in range markets
# and trending away from VWAP in strong trends. Long when price crosses above VWAP in uptrend
# with volume confirmation; short when price crosses below VWAP in downtrend with volume confirmation.
# Weekly trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades.
# Designed for low trade frequency (15-35/year) to minimize whipsaw and capture high-probability moves.

name = "6h_VWAP_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for each day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.zeros_like(close_1d)
    cumulative_tp_vol = np.zeros_like(close_1d)
    cumulative_vol = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        tpv = typical_price_1d[i] * volume_1d[i]
        if i == 0:
            cumulative_tp_vol[i] = tpv
            cumulative_vol[i] = volume_1d[i]
        else:
            cumulative_tp_vol[i] = cumulative_tp_vol[i-1] + tpv
            cumulative_vol[i] = cumulative_vol[i-1] + volume_1d[i]
        
        if cumulative_vol[i] > 0:
            vwap_1d[i] = cumulative_tp_vol[i] / cumulative_vol[i]
        else:
            vwap_1d[i] = typical_price_1d[i]
    
    # Align VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend filter (more responsive than 21)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 50-period EMA
    vol_ema = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price crosses above VWAP in weekly uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > vwap_aligned[i] and       # Price above VWAP
                close[i-1] <= vwap_aligned[i-1] and  # Was at or below VWAP previous bar
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price crosses below VWAP in weekly downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < vwap_aligned[i] and       # Price below VWAP
                  close[i-1] >= vwap_aligned[i-1] and  # Was at or above VWAP previous bar
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below VWAP or trend turns down
            if close[i] < vwap_aligned[i] or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above VWAP or trend turns up
            if close[i] > vwap_aligned[i] or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals