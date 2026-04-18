#!/usr/bin/env python3
"""
1h_4h_1D_Camarilla_R1S1_Breakout_Volume_Selective
Hypothesis: Use daily and 4h Camarilla R1/S1 for directional bias with 1h entry, requiring volume > 1.5x 20-period average and session filter (08-20 UTC). Designed for 15-35 trades/year to minimize fee drag, with volume filter to confirm breakouts. Works in bull/bear via volatility regime filter using 4h ATR.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for primary directional bias
    df_1d = get_htf_data(prices, '1d')
    
    # Get 4h data for volatility filter and secondary confirmation
    df_4h = get_htf_data(prices, '4h')
    
    # Daily calculations for bias
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
    
    # Daily Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1_1d = prev_close + range_1d * 1.1 / 12
    s1_1d = prev_close - range_1d * 1.1 / 12
    
    # 4h ATR for volatility filter (avoid chop)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = np.maximum(high_4h - low_4h, np.absolute(high_4h - np.roll(close_4h, 1)))
    tr2 = np.absolute(np.roll(close_4h, 1) - low_4h)
    tr_4h = np.maximum(tr1, tr2)
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all higher timeframe data to 1h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ATR and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volatility filter: avoid extreme volatility (stop hunting)
        # Use 4h ATR compared to its 50-period average
        atr_ma = pd.Series(atr_4h_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_4h_aligned[i] < atr_ma[i] * 1.5 if not np.isnan(atr_ma[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above daily R1 with volume and volatility filter during session
            if close[i] > r1_1d_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below daily S1 with volume and volatility filter during session
            elif close[i] < s1_1d_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price returns below daily R1 or volatility filter fails or outside session
            if close[i] < r1_1d_aligned[i] or not vol_filter or not in_session:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns above daily S1 or volatility filter fails or outside session
            if close[i] > s1_1d_aligned[i] or not vol_filter or not in_session:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1D_Camarilla_R1S1_Breakout_Volume_Selective"
timeframe = "1h"
leverage = 1.0