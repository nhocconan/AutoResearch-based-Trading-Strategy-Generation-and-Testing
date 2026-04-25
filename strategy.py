#!/usr/bin/env python3
"""
1h Camarilla Pivot H3L3 Breakout + 4h EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3L3 levels act as significant intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and aligned 4h EMA34 trend 
capture momentum moves. The 4h EMA34 filter ensures we trade with the higher timeframe 
trend, reducing false breakouts. Designed for 1h timeframe with 60-150 total trades 
over 4 years (15-37/year) to balance opportunity and fee drag. Session filter (08-20 UTC) 
reduces noise trades outside active market hours.
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
    
    # Get 4h data for EMA34 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need at least 34 bars for EMA34
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivots
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    # Based on previous day's range
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (no extra delay needed for pivots)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and pivots
    start_idx = max(34, 20, 1)  # EMA34 needs 34, volume MA needs 20, pivots need 1 day
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 4h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend
            long_breakout = (curr_close > h3) and volume_confirm and uptrend
            # Short: price breaks below L3 with volume confirmation in downtrend
            short_breakout = (curr_close < l3) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below H3 OR EMA34 trend turns down
            if curr_close < h3 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above L3 OR EMA34 trend turns up
            if curr_close > l3 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0