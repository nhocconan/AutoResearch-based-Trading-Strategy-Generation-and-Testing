#!/usr/bin/env python3
"""
6h Weekly Donchian(20) Breakout + 1d Volume Spike + 1w EMA34 Trend Filter
Hypothesis: Weekly trend (1w EMA34) filters breakout direction from 6h Donchian channels, with volume confirmation on 1d to ensure institutional participation. Designed for 6h timeframe to capture multi-day momentum while avoiding overtrading and respecting BTC/ETH mean-reverting bear markets (2025+) by requiring weekly alignment.
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
    
    # Get 1w data for EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for volume confirmation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_1d = pd.Series(df_1d['volume'])
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for dynamic stop on 6h data
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Donchian channels (20-period) on 6h data
    high_6h = pd.Series(high)
    low_6h = pd.Series(low)
    donchian_high_20 = high_6h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_6h.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for weekly EMA34, daily volume MA, ATR, Donchian
    start_idx = max(34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i])):
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
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_val = atr[i]
        donchian_high = donchian_high_20[i]
        donchian_low = donchian_low_20[i]
        
        # Trend filter: price relative to 1w EMA34 (multi-week trend)
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average (institutional participation)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Donchian levels
            # Long: price breaks above Donchian high with volume confirmation in uptrend
            long_breakout = (curr_close > donchian_high) and volume_confirm and uptrend
            # Short: price breaks below Donchian low with volume confirmation in downtrend
            short_breakout = (curr_close < donchian_low) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below Donchian low OR 2.5*ATR trailing stop OR weekly EMA34 trend turns down
            if curr_close < donchian_low or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Donchian high OR 2.5*ATR trailing stop OR weekly EMA34 trend turns up
            if curr_close > donchian_high or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchianBreakout_1dVolumeSpike_1wEMA34Trend"
timeframe = "6h"
leverage = 1.0