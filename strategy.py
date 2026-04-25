#!/usr/bin/env python3
"""
1h Volume Spike + 4h Donchian Breakout + 1d EMA50 Trend Filter
Hypothesis: Combine 4h Donchian breakouts for structure, 1h volume spikes for momentum confirmation, and 1d EMA50 for trend filter. 
This multi-timeframe approach reduces false breakouts by requiring alignment across timeframes. 
Session filter (08-20 UTC) avoids low-liquidity periods. Designed for 1h timeframe targeting 15-35 trades/year.
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
    if len(df_4h) < 20:  # Need at least 20 periods for Donchian
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 periods for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA50, and volume MA
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above 4h Donchian high with volume confirmation in uptrend
            long_breakout = (curr_close > donchian_high_val) and volume_confirm and uptrend
            # Short: price breaks below 4h Donchian low with volume confirmation in downtrend
            short_breakout = (curr_close < donchian_low_val) and volume_confirm and downtrend
            
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
            # Exit long: price closes below 4h Donchian low OR EMA50 trend turns down
            if curr_close < donchian_low_val or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above 4h Donchian high OR EMA50 trend turns up
            if curr_close > donchian_high_val or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_DonchianBreakout_1dEMA50Trend"
timeframe = "1h"
leverage = 1.0