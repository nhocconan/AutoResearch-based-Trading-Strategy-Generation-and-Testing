#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend direction.
- Donchian Channel: 20-period high/low breakouts for trend following.
- Trend Filter: 1d EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 4h volume > 1.8 * 20-period average 4h volume to confirm strong momentum.
- Entry: Long when close > Donchian High(20) AND close > 1d EMA50 AND volume spike.
         Short when close < Donchian Low(20) AND close < 1d EMA50 AND volume spike.
- Exit: Opposite Donchian break (long exits when close < Donchian Low(10), short exits when close > Donchian High(10)).
- Signal size: 0.25 discrete to minimize fee drag.
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
    
    # Calculate Donchian Channels (20-period for entry, 10-period for exit)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i]) or np.isnan(donchian_high_10[i]) or
            np.isnan(donchian_low_10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        donch_high_20 = donchian_high_20[i]
        donch_low_20 = donchian_low_20[i]
        donch_high_10 = donchian_high_10[i]
        donch_low_10 = donchian_low_10[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_dc20 = curr_close > donch_high_20
        broke_below_dc20 = curr_close < donch_low_20
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break (using 10-period for smoother exit)
        if position != 0:
            # Exit long: close breaks below Donchian Low(10)
            if position == 1:
                if curr_close < donch_low_10:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above Donchian High(10)
            elif position == -1:
                if curr_close > donch_high_10:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above DC20 AND above EMA50 AND volume spike
            long_condition = broke_above_dc20 and above_ema and volume_spike
            
            # Short: break below DC20 AND below EMA50 AND volume spike
            short_condition = broke_below_dc20 and below_ema and volume_spike
            
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