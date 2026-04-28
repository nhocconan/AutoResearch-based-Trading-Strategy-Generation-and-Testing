#!/usr/bin/env python3
"""
4h_VWAP_Deviation_12hTrend_VolumeFilter
Hypothesis: Mean-reversion from VWAP deviation with 12h trend filter and volume confirmation.
In strong trends (12h EMA50), price often reverts to VWAP, providing low-risk entries.
Works in both bull and bear markets by trading with the 12h trend.
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
    
    # Get 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Calculate cumulative VWAP (reset daily)
    vwap = np.full(n, np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    
    # Group by date for daily VWAP reset
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        for i in idxs:
            cum_tpv += tpv[i]
            cum_vol += volume[i]
            if cum_vol > 0:
                vwap[i] = cum_tpv / cum_vol
        # Reset at end of day
        cum_tpv = 0.0
        cum_vol = 0.0
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or vwap[i] == 0):
            signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP as percentage
        deviation = (close[i] - vwap[i]) / vwap[i]
        
        # Trend direction from 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: >1.5x 20-period MA
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Mean-reversion entries: price deviates from VWAP but trend supports reversion
        long_entry = vol_confirm and trend_down and (deviation < -0.015)  # 1.5% below VWAP in downtrend
        short_entry = vol_confirm and trend_up and (deviation > 0.015)   # 1.5% above VWAP in uptrend
        
        # Exit when price returns to VWAP or trend reverses
        long_exit = (deviation > -0.005) or (not trend_down)  # Near VWAP or trend change
        short_exit = (deviation < 0.005) or (not trend_up)    # Near VWAP or trend change
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_VWAP_Deviation_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0