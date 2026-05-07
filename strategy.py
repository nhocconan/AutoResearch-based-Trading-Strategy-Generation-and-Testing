#!/usr/bin/env python3
# 1D_VWAP_RollingBreakout_WeeklyTrend_VolumeConfirm
# Hypothesis: Breakout above/below rolling VWAP (50-period) on 1d timeframe, filtered by weekly trend (EMA 50) and volume spike (current volume > 1.8x 20-day average). Uses 1d as primary timeframe and 1w for trend filter. VWAP acts as dynamic support/resistance; breakouts with volume confirmation capture institutional moves. Weekly EMA filter ensures trades align with higher-timeframe trend, reducing whipsaws. Works in bull/bear by following weekly trend. Designed for low trade frequency (10-20/year) to minimize fee drag.

name = "1D_VWAP_RollingBreakout_WeeklyTrend_VolumeConfirm"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA 50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate rolling VWAP (50-period) on 1d
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = pd.Series(pv).rolling(window=50, min_periods=50).sum().values
    cum_vol = pd.Series(volume).rolling(window=50, min_periods=50).sum().values
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Volume spike: current volume > 1.8x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have VWAP and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above VWAP + up-trend (close > weekly EMA50) + volume spike
            if (close[i] > vwap[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below VWAP + down-trend (close < weekly EMA50) + volume spike
            elif (close[i] < vwap[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below VWAP or weekly EMA50 (mean reversion)
            if close[i] < vwap[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above VWAP or weekly EMA50 (mean reversion)
            if close[i] > vwap[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals