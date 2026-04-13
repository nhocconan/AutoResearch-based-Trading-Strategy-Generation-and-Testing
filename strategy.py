#!/usr/bin/env python3
"""
4h_1d_VWAP_Bounce_With_Trend
Hypothesis: Combines daily VWAP as dynamic support/resistance with 4h EMA trend filter.
In trending markets, price often pulls back to VWAP before continuing trend. 
Enters long when price bounces above VWAP in uptrend (price > EMA50), short when 
price rejects VWAP in downtrend (price < EMA50). Uses volume confirmation to 
filter false signals. Works in both bull and bear markets by trading with the 
trend on pullbacks to dynamic value area. Target: 20-50 trades/year on 4h.
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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = typical_price_1d * volume_1d
    vwap_denominator = volume_1d
    
    # Cumulative VWAP (resets daily)
    cum_vwap_num = np.nancumsum(vwap_numerator)
    cum_vwap_den = np.nancumsum(vwap_denominator)
    vwap_1d = np.where(cum_vwap_den != 0, cum_vwap_num / cum_vwap_den, typical_price_1d)
    
    # Reset VWAP at daily boundaries (when date changes)
    dates = pd.to_datetime(df_1d.index).date if hasattr(df_1d.index, 'date') else \
            pd.to_datetime(df_1d.index).date
    if len(df_1d) > 0:
        date_changes = np.concatenate(([True], dates[1:] != dates[:-1]))
        for i in range(1, len(date_changes)):
            if date_changes[i]:
                cum_vwap_num[i] = vwap_numerator[i]
                cum_vwap_den[i] = vwap_denominator[i]
                vwap_1d[i] = typical_price_1d[i]
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all signals to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(vwap_1d_aligned[i]) or \
           np.isnan(ema_50_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
            volume_confirm = volume[i] > (vol_ma_20 * 1.2)
        else:
            volume_confirm = False
        
        # Entry conditions
        if volume_confirm:
            # Long: price above VWAP and above EMA50 (uptrend bounce)
            if close[i] > vwap_1d_aligned[i] and close[i] > ema_50_4h_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short: price below VWAP and below EMA50 (downtrend rejection)
            elif close[i] < vwap_1d_aligned[i] and close[i] < ema_50_4h_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold current position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No volume confirmation - hold or flatten
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_VWAP_Bounce_With_Trend"
timeframe = "4h"
leverage = 1.0