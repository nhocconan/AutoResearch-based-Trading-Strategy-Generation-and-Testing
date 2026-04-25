#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume spike confirmation.
Targets 7-25 trades/year by requiring: 1) price breaks weekly Donchian high/low (strong weekly breakout),
2) aligned with weekly EMA34 trend, 3) volume > 2.0x 20-period average. This strategy focuses on
1d timeframe to minimize fee drag while capturing significant moves in both bull and bear markets.
Weekly trend filter ensures we only trade with the dominant higher-timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w data for Donchian channels (20-period high/low) (loaded ONCE)
    # Donchian high = max(high, 20), Donchian low = min(low, 20)
    dh_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 1w levels to 1d timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1w EMA34 (34) and Donchian (20)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment
            # Long breakout: price breaks above weekly Donchian high with uptrend and volume confirmation
            long_breakout = (curr_close > dh_20_aligned[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below weekly Donchian low with downtrend and volume confirmation
            short_breakout = (curr_close < dl_20_aligned[i]) and downtrend and volume_confirm[i]
            
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
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below weekly Donchian low (mean reversion) or trend changes
            if curr_close < dl_20_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above weekly Donchian high (mean reversion) or trend changes
            if curr_close > dh_20_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0