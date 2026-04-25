#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian breakouts capture strong moves. 12h EMA50 filters trend direction.
Volume spike confirms breakout strength. ATR-based stop manages risk.
Works in bull (breakouts up) and bear (breakouts down) via trend filter.
Target: 20-50 trades/year per symbol (<200 total over 4 years).
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
    
    # Get 12h data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels on 4h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period ATR for stop loss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr_20 = np.full(n, np.nan)
    for i in range(20, n):
        atr_20[i] = np.mean(tr[i-19:i+1])
    
    # Calculate 20-period volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 20)  # Donchian, ATR, volume MA all need 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        atr_val = atr_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout entries
            # Long: price breaks above upper Donchian + uptrend + volume confirmation
            long_breakout = curr_close > upper_donchian
            long_entry = long_breakout and (curr_close > ema_trend) and volume_confirm
            
            # Short: price breaks below lower Donchian + downtrend + volume confirmation
            short_breakout = curr_close < lower_donchian
            short_entry = short_breakout and (curr_close < ema_trend) and volume_confirm
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below lower Donchian OR 2*ATR stoploss
            if (curr_close < lower_donchian or 
                curr_close < entry_price - 2.0 * atr_at_entry):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price breaks above upper Donchian OR 2*ATR stoploss
            if (curr_close > upper_donchian or 
                curr_close > entry_price + 2.0 * atr_at_entry):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0