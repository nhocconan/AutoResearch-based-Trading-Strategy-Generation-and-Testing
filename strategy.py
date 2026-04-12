#/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_trend
Hypothesis: 4-hour strategy using Camarilla pivot levels from daily timeframe for entry/exit, with volume confirmation and trend filtering.
Uses daily Camarilla levels (H4, L4) for breakout entries, volume > 1.5x 20-period average for confirmation, and EMA21 trend filter.
Designed to work in both bull and bear markets by only taking breakouts aligned with daily EMA21 trend.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Camarilla levels
    # Camarilla: H4 = close + 1.5 * (high - low), L4 = close - 1.5 * (high - low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Daily EMA21 for trend filter
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend determination: EMA21 > price = uptrend, EMA21 < price = downtrend
        uptrend = ema21_1d_aligned[i] > close[i]
        downtrend = ema21_1d_aligned[i] < close[i]
        
        # Entry conditions
        if vol_confirmed:
            if uptrend and close[i] > camarilla_h4_aligned[i] and position != 1:
                # Long breakout above H4 in uptrend
                position = 1
                signals[i] = 0.25
            elif downtrend and close[i] < camarilla_l4_aligned[i] and position != -1:
                # Short breakdown below L4 in downtrend
                position = -1
                signals[i] = -0.25
        # Exit conditions: opposite Camarilla level touch or trend reversal
        elif position == 1:
            if close[i] < camarilla_l4_aligned[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > camarilla_h4_aligned[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0