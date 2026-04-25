#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + Volume Spike + 1d EMA34 Trend
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance derived from prior day's range.
Breakouts with volume confirmation (>2x 20-bar volume MA) and aligned 1d EMA34 trend capture
strong momentum moves while avoiding false breakouts. Works in bull/bear markets by using
1d EMA34 as trend filter. Discrete position sizing (0.25) minimizes fee churn.
Target: 75-200 trades over 4 years to balance opportunity and fee drag.
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
    
    # Get daily data for Camarilla pivot calculation and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA34 + 1 for previous day
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from PREVIOUS day's OHLC (to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's OHLC for Camarilla calculation
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # First day has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla levels: H3/L3 for entry, H4/L4 for stop reference
    # H3 = prev_close + 1.1*(prev_high - prev_low)/2
    # L3 = prev_close - 1.1*(prev_high - prev_low)/2
    camarilla_h3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and aligned data
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3) and volume_confirm and uptrend
            # Short: price breaks below L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below L3 OR EMA34 trend turns down
            if curr_close < camarilla_l3 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 trend turns up
            if curr_close > camarilla_h3 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0