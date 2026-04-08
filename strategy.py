#!/usr/bin/env python3
"""
6h_1w_1d_vwap_reversion_v1
Hypothesis: Mean reversion to VWAP on 6h timeframe with weekly bias filter.
- Calculate 1d VWAP from daily data
- Use weekly trend (1w close vs open) as bias filter
- Enter long when price < VWAP and weekly bullish, short when price > VWAP and weekly bearish
- Exit when price crosses VWAP or weekly bias reverses
- VWAP acts as dynamic equilibrium price in ranging markets
- Weekly bias prevents counter-trend trades in strong trends
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly bias: bullish if weekly close > open, bearish if close < open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    
    # Forward fill weekly bias to get current week's bias
    weekly_bullish_series = pd.Series(weekly_bullish)
    weekly_bearish_series = pd.Series(weekly_bearish)
    weekly_bullish_ffilled = weekly_bullish_series.ffill().values
    weekly_bearish_ffilled = weekly_bearish_series.ffill().values
    
    # Align weekly bias to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_ffilled)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_ffilled)
    
    # Get daily data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP: typical price * volume / cumulative volume
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    pv = typical_price * df_1d['volume'].values
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(df_1d['volume'].values)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Forward fill VWAP
    vwap_series = pd.Series(vwap)
    vwap_ffilled = vwap_series.ffill().values
    
    # Align VWAP to 6h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_ffilled)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vwap_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above VWAP or weekly bias turns bearish
            if close[i] >= vwap_aligned[i] or weekly_bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price crosses below VWAP or weekly bias turns bullish
            if close[i] <= vwap_aligned[i] or weekly_bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price below VWAP with weekly bullish bias
            if close[i] < vwap_aligned[i] and weekly_bullish_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price above VWAP with weekly bearish bias
            elif close[i] > vwap_aligned[i] and weekly_bearish_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals