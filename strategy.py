#!/usr/bin/env python3
"""
1h Volume Spike + 4h Donchian Breakout + 1d EMA34 Trend Filter
Hypothesis: In 1h timeframe, volume spikes confirm institutional participation. 
Breakout of 4h Donchian channels (20-period) captures momentum. 
1d EMA34 filter ensures we only trade in direction of higher timeframe trend.
Works in bull (long on upper break) and bear (short on lower break). 
Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year on 1h.
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
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian upper (20-period high)
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # 4h Donchian lower (20-period low)
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA34, volume MA
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        donchian_high = donchian_high_4h_aligned[i]
        donchian_low = donchian_low_4h_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0 and in_session:
            # Look for entry signals
            # Long: price > 4h Donchian high, above 1d EMA34, volume confirmation
            long_entry = (curr_close > donchian_high) and (curr_close > ema_34_val) and volume_confirm
            # Short: price < 4h Donchian low, below 1d EMA34, volume confirmation
            short_entry = (curr_close < donchian_low) and (curr_close < ema_34_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below 1d EMA34 OR Donchian low (trailing)
            if curr_close < ema_34_val or curr_low < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above 1d EMA34 OR Donchian high (trailing)
            if curr_close > ema_34_val or curr_high > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_DonchianBreakout_1dEMA34Trend_Session"
timeframe = "1h"
leverage = 1.0