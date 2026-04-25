#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA34 Trend + Volume Spike Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Using 12h EMA34 as trend filter ensures we only trade in direction of higher timeframe trend.
Volume spike confirmation reduces false breakouts. 
ATR-based stop loss manages risk. 
Designed for 4h timeframe with tight entry conditions to target 75-200 trades over 4 years.
Works in both bull and bear markets by following the 12h trend direction.
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
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = pd.Series(df_12h['close'])
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(14) for stop loss and position sizing (4h)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34, ATR, and volume MA
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Donchian breakout levels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for long breakout: price breaks above Donchian high 
            # AND price above 12h EMA34 (uptrend filter) AND volume confirmation
            long_breakout = (curr_close > donchian_high) and (curr_close > ema_34_val) and volume_confirm
            
            # Look for short breakdown: price breaks below Donchian low 
            # AND price below 12h EMA34 (downtrend filter) AND volume confirmation
            short_breakout = (curr_close < donchian_low) and (curr_close < ema_34_val) and volume_confirm
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Long position: exit if price closes below Donchian low OR 2*ATR stop loss
            if curr_close < donchian_low or curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price closes above Donchian high OR 2*ATR stop loss
            if curr_close > donchian_high or curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0