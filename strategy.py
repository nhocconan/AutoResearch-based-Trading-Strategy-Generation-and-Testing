#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout + 4h EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) from 4h provide intraday support/resistance. 
Breakout above H3 with 4h EMA34 uptrend and volume spike captures momentum in bull markets.
Breakdown below L3 with 4h EMA34 downtrend and volume spike captures momentum in bear markets.
Session filter (08-20 UTC) reduces noise. Target 15-35 trades/year on 1h to avoid fee drag.
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
    
    # Get 4h data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivots on 4h
    # Pivot point = (high + low + close) / 3
    pp = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    # Ranges
    range_hl = df_4h['high'] - df_4h['low']
    # Camarilla levels
    h3 = pp + (range_hl * 1.1 / 4)
    l3 = pp - (range_hl * 1.1 / 4)
    h4 = pp + (range_hl * 1.1 / 2)
    l4 = pp - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4.values)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price above H3, EMA34 uptrend, volume confirmation
            long_entry = (curr_close > h3_val) and (curr_close > ema_34_val) and volume_confirm
            # Short: price below L3, EMA34 downtrend, volume confirmation
            short_entry = (curr_close < l3_val) and (curr_close < ema_34_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: price closes below L3 OR EMA34 downtrend
            if curr_close < l3_val or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit conditions: price closes above H3 OR EMA34 uptrend
            if curr_close > h3_val or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0