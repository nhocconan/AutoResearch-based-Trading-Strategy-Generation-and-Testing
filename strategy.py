#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Target_V1
Hypothesis: Combines daily Camarilla pivot levels with 12h breakouts and volume confirmation.
In trending markets, price breaks above/below key Camarilla levels (H4/L4) with volume expansion.
During ranging markets (Choppiness Index > 61.8), avoids false breakouts.
Target: 15-35 trades/year on 12h (60-140 total over 4 years).
Works in both bull and bear markets by filtering trades with regime and volume.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # where C = (H+L+C)/3 (typical price)
    # Using previous day's data to avoid lookahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot using previous day's OHLC
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    cam_h4 = typical_price + 1.5 * (high_1d - low_1d)
    cam_l4 = typical_price - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (using previous day's values)
    cam_h4_aligned = align_htf_to_ltf(prices, df_1d, cam_h4)
    cam_l4_aligned = align_htf_to_ltf(prices, df_1d, cam_l4)
    
    # Calculate 12h Choppiness Index for regime filtering
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr.rolling(window=period, min_periods=period).sum() / (max_high - min_low)) / np.log10(period)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Chop > 61.8 = ranging market (avoid breakouts)
    chop_filter = chop < 61.8  # only trade when NOT ranging
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(cam_h4_aligned[i]) or \
           np.isnan(cam_l4_aligned[i]) or \
           np.isnan(chop_filter[i]) or \
           np.isnan(volume_expansion[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > cam_h4_aligned[i]) and volume_expansion[i] and chop_filter[i]
        short_entry = (close[i] < cam_l4_aligned[i]) and volume_expansion[i] and chop_filter[i]
        
        # Exit conditions: reverse signal or loss of momentum
        long_exit = (position == 1) and (close[i] < cam_l4_aligned[i])
        short_exit = (position == -1) and (close[i] > cam_h4_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif long_exit or short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Breakout_Target_V1"
timeframe = "12h"
leverage = 1.0