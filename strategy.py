#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend direction (long-term trend filter).
- Donchian Channel: 20-period high/low on 12h for breakout logic.
- Trend Filter: 1w EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 12h volume > 1.8 * 20-period average 12h volume to confirm strong momentum.
- Entry: Long when close > Donchian High(20) AND close > 1w EMA50 AND volume spike.
         Short when close < Donchian Low(20) AND close < 1w EMA50 AND volume spike.
- Exit: Opposite Donchian break (long exits when close < Donchian Low(10), short exits when close > Donchian High(10)).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with weekly trend while filtering chop/whipsaws.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20 for entry, 10 for exit)
    # Donchian High(20): highest high of last 20 periods
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian High(10) and Low(10) for exits
    donch_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for weekly EMA, 20 for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_high_20[i]) or
            np.isnan(donch_low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_level = ema_50_1w_aligned[i]
        donch_high_20_level = donch_high_20[i]
        donch_low_20_level = donch_low_20[i]
        donch_high_10_level = donch_high_10[i]
        donch_low_10_level = donch_low_10[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions (20-period)
        broke_above_donch_high = curr_close > donch_high_20_level
        broke_below_donch_low = curr_close < donch_low_20_level
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break (10-period for quicker exit)
        if position != 0:
            # Exit long: close breaks below Donchian Low(10)
            if position == 1:
                if curr_close < donch_low_10_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above Donchian High(10)
            elif position == -1:
                if curr_close > donch_high_10_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above Donchian High(20) AND above EMA50 AND volume spike
            long_condition = broke_above_donch_high and above_ema and volume_spike
            
            # Short: break below Donchian Low(20) AND below EMA50 AND volume spike
            short_condition = broke_below_donch_low and below_ema and volume_spike
            
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

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0