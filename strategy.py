#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 1h with 4h EMA50 trend filter and volume confirmation.
Uses 4h for signal direction and 1h for precise entry timing to reduce false breakouts.
Session filter (08-20 UTC) avoids low-liquidity periods. Target: 15-30 trades/year per symbol.
Works in bull markets (breakouts with trend) and bear markets (fade breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla H3/L3 levels (from daily chart)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_h3_1d = typical_price_1d + (range_1d * 1.1 / 4.0)
    camarilla_l3_1d = typical_price_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: 1h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA50)
        htf_4h_bullish = close[i] > ema_50_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + 4h uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_4h_bullish and volume_spike[i]
            
            # Short setup: price breaks below Camarilla L3 + 4h downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_4h_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches Camarilla L3 (stop) OR 4h trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches Camarilla H3 (stop) OR 4h trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0