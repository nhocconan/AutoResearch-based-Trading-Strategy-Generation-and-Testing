#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike + ATR(14) Stop
Hypothesis: 12h Donchian breakouts capture momentum, filtered by 1w EMA34 trend for multi-timeframe alignment and volume spike (>2x 20-period MA) for confirmation. ATR trailing stop manages risk. Designed for low trade frequency (target: 50-150/4 years) to minimize fee drag. Works in bull/bear via weekly trend filter.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 weeks for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for trailing stop (12h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Donchian channels for 12h (using previous 20 periods)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need enough for EMA34, volume MA, ATR, and Donchian
    start_idx = max(34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Donchian channels
            # Long: price breaks above Donchian high with volume confirmation in uptrend
            long_breakout = (curr_close > donchian_high_val) and volume_confirm and uptrend
            # Short: price breaks below Donchian low with volume confirmation in downtrend
            short_breakout = (curr_close < donchian_low_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: 3 * ATR below highest since entry
            trailing_stop = highest_since_entry - 3.0 * atr_val
            # Exit conditions: price closes below Donchian low OR trailing stop hit OR EMA34 trend turns down
            if curr_close < donchian_low_val or curr_close < trailing_stop or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: 3 * ATR above lowest since entry
            trailing_stop = lowest_since_entry + 3.0 * atr_val
            # Exit conditions: price closes above Donchian high OR trailing stop hit OR EMA34 trend turns up
            if curr_close > donchian_high_val or curr_close > trailing_stop or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wEMA34_Trend_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0