#!/usr/bin/env python3
"""
1h EMA Cross + 4h Donchian Breakout + Volume Spike + Session Filter
Hypothesis: On 1h timeframe, use EMA(9,21) for entry timing, 4h Donchian(20) for trend direction,
and volume spike for confirmation. Only trade during 08-20 UTC to avoid low-liquidity hours.
Designed for BTC/ETH with 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Works in bull/bear markets: 4h Donchian breaks capture strong moves, EMA cross provides precise timing.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower = rolling max(high,20)/min(low,20) on 4h
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (waits for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Calculate 1h EMA(9) and EMA(21) for entry timing
    close_s = pd.Series(close)
    ema_9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA21, Donchian, volume MA
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_9_val = ema_9[i]
        ema_21_val = ema_21[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 4h Donchian channels
        # Uptrend: price above Donchian upper (breakout)
        # Downtrend: price below Donchian lower (breakdown)
        uptrend = curr_close > donchian_high
        downtrend = curr_close < donchian_low
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # EMA cross signals
        ema_bullish = ema_9_val > ema_21_val
        ema_bearish = ema_9_val < ema_21_val
        
        if position == 0:
            # Look for entry signals
            # Long: EMA bullish cross + price above Donchian high + volume confirmation
            long_entry = ema_bullish and uptrend and volume_confirm
            # Short: EMA bearish cross + price below Donchian low + volume confirmation
            short_entry = ema_bearish and downtrend and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: EMA bearish cross OR price breaks below Donchian low
            if ema_bearish or curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA bullish cross OR price breaks above Donchian high
            if ema_bullish or curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Cross_4hDonchian_Breakout_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0