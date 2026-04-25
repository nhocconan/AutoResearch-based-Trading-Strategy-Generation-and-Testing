#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend direction and Donchian(20) channels from prior day.
- Donchian Channels: Upper = 20-period high, Lower = 20-period low from prior 1d OHLC.
- Trend Filter: 1d EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 4h volume > 1.8 * 20-period average 4h volume to confirm strong momentum.
- Entry: Long when close > Upper Channel AND close > 1d EMA50 AND volume spike.
         Short when close < Lower Channel AND close < 1d EMA50 AND volume spike.
- Exit: Opposite Donchian break (long exits when close < Lower Channel, short exits when close > Upper Channel).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with daily trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian channels (20-period) from prior day OHLC
    # Shifted to avoid look-ahead: use prior day's data for today's channels
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d_shifted = df_1d['close'].shift(1).values
    
    # Donchian Upper = 20-period high, Lower = 20-period low from prior day
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (waits for 1d bar close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need 50 for EMA, 20 for Donchian, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_channel = donchian_upper_aligned[i]
        lower_channel = donchian_lower_aligned[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_channel
        broke_below_lower = curr_close < lower_channel
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower channel
            if position == 1:
                if curr_close < lower_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper channel
            elif position == -1:
                if curr_close > upper_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper channel AND above EMA50 AND volume spike
            long_condition = broke_above_upper and above_ema and volume_spike
            
            # Short: break below lower channel AND below EMA50 AND volume spike
            short_condition = broke_below_lower and below_ema and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0