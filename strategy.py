#!/usr/bin/env python3
"""
1d_VWAP_Trend_Breakout
Hypothesis: VWAP breakout with daily trend filter (weekly EMA) and volume confirmation.
Works in bull markets (breakout above VWAP in uptrend) and bear markets (breakdown below VWAP in downtrend).
Targets 10-20 trades/year to minimize fee drag while capturing significant moves.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below EMA20
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Price relative to VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: VWAP breakout in direction of weekly trend
        long_entry = vol_confirm and weekly_uptrend and above_vwap
        short_entry = vol_confirm and weekly_downtrend and below_vwap
        
        # Exit logic: opposite VWAP cross or trend change
        long_exit = (below_vwap) or (not weekly_uptrend)
        short_exit = (above_vwap) or (not weekly_downtrend)
        
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

name = "1d_VWAP_Trend_Breakout"
timeframe = "1d"
leverage = 1.0