#!/usr/bin/env python3
"""
4h_VWAP_Cross_1dTrend_VolumeFilter
Hypothesis: Price crossing VWAP on 4h with 1d EMA trend filter and volume confirmation.
Works in both bull and bear markets by trading with the daily trend.
Targets ~30 trades/year to minimize fee drag.
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate VWAP (Volume Weighted Average Price) for 4h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_21_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA21
        trend_up = close[i] > ema_21_1d_aligned[i]
        trend_down = close[i] < ema_21_1d_aligned[i]
        
        # Volume confirmation: >1.8x 20-period MA
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # VWAP cross conditions
        vwap_cross_up = close[i] > vwap[i] and close[i-1] <= vwap[i-1]
        vwap_cross_down = close[i] < vwap[i] and close[i-1] >= vwap[i-1]
        
        # Entry logic: VWAP cross with trend and volume
        long_entry = vwap_cross_up and trend_up and vol_confirm
        short_entry = vwap_cross_down and trend_down and vol_confirm
        
        # Exit logic: opposite VWAP cross or trend reversal
        long_exit = vwap_cross_down or (not trend_up)
        short_exit = vwap_cross_up or (not trend_down)
        
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

name = "4h_VWAP_Cross_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0