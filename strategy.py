#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_Breakout_Volume_V1
Hypothesis: Use 1D Camarilla R1/S1 breakouts on 12h timeframe with strict volume confirmation (2x average) to capture strong directional moves while minimizing trades. Works in bull/bear via volatility filter (avoid chop) and session focus (08-20 UTC). Target: 15-30 trades/year per symbol to avoid fee drag.
"""

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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Volatility filter: use ATR(20) to avoid choppy markets
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volatility filter: avoid extreme volatility (stop hunting)
        vol_ma_long = pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_20_aligned[i] < vol_ma_long[i] * 2 if not np.isnan(vol_ma_long[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility filter during session
            if close[i] > r1_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility filter during session
            elif close[i] < s1_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or volatility spike or outside session
            if close[i] < r1_aligned[i] or not vol_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or volatility spike or outside session
            if close[i] > s1_aligned[i] or not vol_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_R1S1_Breakout_Volume_V1"
timeframe = "12h"
leverage = 1.0