#!/usr/bin/env python3
"""
12h Multi-Timeframe Trend Reversal with Volume Confirmation
Hypothesis: Price reversals at 1-day high/low levels with volume confirmation and 1-week trend filter work in both bull and bear markets. Uses 1-week EMA for trend direction and 1-day high/low for reversal signals. Volume spike confirms conviction. Designed for 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for reversal levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA for trend direction (50-period)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily high and low for reversal levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume spike detection (2x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema_1w_aligned[i]
        daily_high_level = high_1d_aligned[i]
        daily_low_level = low_1d_aligned[i]
        
        if position == 0:
            # Look for reversal signals with volume confirmation
            # Long setup: price near daily low in uptrend with volume spike
            if price <= daily_low_level * 1.002 and price > trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price near daily high in downtrend with volume spike
            elif price >= daily_high_level * 0.998 and price < trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until reversal or trend change
            signals[i] = 0.25
            # Exit: price reaches daily high or trend turns down
            if price >= daily_high_level or price < trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal or trend change
            signals[i] = -0.25
            # Exit: price reaches daily low or trend turns up
            if price <= daily_low_level or price > trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_MultiTimeframe_TrendReversal_VolumeConfirm"
timeframe = "12h"
leverage = 1.0