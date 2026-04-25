#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA20 Trend + Volume Spike + Session Filter (08-20 UTC)
Hypothesis: Camarilla pivot breakouts on 1h capture intraday momentum, filtered by 4h EMA20 trend and volume confirmation. Session filter reduces noise. Designed for 1h timeframe to avoid overtrading while working in both bull (long breakouts) and bear (short breakouts) via symmetric logic. Target: 60-150 total trades over 4 years (15-37/year).
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
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h Camarilla pivots (based on previous day's high/low/close)
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily high, low, close for pivot points
    high_1d = pd.Series(df_1d['high']).values
    low_1d = pd.Series(df_1d['low']).values
    close_1d = pd.Series(df_1d['close']).values
    
    # Camarilla levels: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA20, volume MA, and Camarilla
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_20_val = ema_20_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        camarilla_high = camarilla_high_aligned[i]
        camarilla_low = camarilla_low_aligned[i]
        
        # Trend filter: price relative to 4h EMA20
        uptrend = curr_close > ema_20_val
        downtrend = curr_close < ema_20_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla levels
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_high) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_low) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: price closes below Camarilla L3 OR EMA20 trend turns down
            if curr_close < camarilla_low or curr_close < ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit conditions: price closes above Camarilla H3 OR EMA20 trend turns up
            if curr_close > camarilla_high or curr_close > ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0