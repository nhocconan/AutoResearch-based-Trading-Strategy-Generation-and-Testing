#!/usr/bin/env python3

# Hypothesis: 6h timeframe with 1-day Volume-Weighted Average Price (VWAP) and 1-week Exponential Moving Average (EMA).
# Long when price > 1d VWAP and > 1w EMA34; short when price < 1d VWAP and < 1w EMA34.
# VWAP acts as dynamic intraday support/resistance, EMA34 provides weekly trend filter.
# Volume confirmation ensures trades occur with institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
# Works in bull markets (trend following) and bear markets (mean reversion to VWAP in range).

name = "6h_VWAP_EMA34_Trend_Volume"
timeframe = "6h"
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
    
    # Calculate 1d VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    # Reset VWAP at start of each day (00:00 UTC)
    # Assuming 6h bars: 4 bars per day
    days = np.arange(len(prices)) // 4
    vwap = np.where(np.diff(np.concatenate(([days[0]], days))) != 0, np.nan, vwap)
    # Forward fill within day
    for i in range(1, len(vwap)):
        if np.isnan(vwap[i]):
            vwap[i] = vwap[i-1]
    
    # Get 1d data for additional filters (optional)
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP and above 1w EMA34 + volume
            if close[i] > vwap[i] and close[i] > ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP and below 1w EMA34 + volume
            elif close[i] < vwap[i] and close[i] < ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or trend reversal
            if close[i] <= vwap[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or trend reversal
            if close[i] >= vwap[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals