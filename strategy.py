#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction.
- Donchian Channel: identifies breakout points from 20-period high/low.
- Entry: Long when price breaks above Donchian upper band AND 12h EMA50 is rising AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian lower band AND 12h EMA50 is falling AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout (price crosses back below upper band for longs, above lower band for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves.
- Volume confirmation ensures breakout legitimacy.
- 12h EMA50 trend filter ensures trades are in direction of higher timeframe trend.
- Works in both bull and bear markets as it captures volatility expansion after contraction.
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
    
    # Calculate 4h Donchian Channel (20-period)
    if n < 20:
        return np.zeros(n)
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume average for confirmation (20-period)
    if n < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below upper band for longs, above lower band for shorts
        if position != 0:
            # Exit long: price crosses below upper band
            if position == 1:
                if curr_close < donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above lower band
            elif position == -1:
                if curr_close > donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with 12h EMA50 trend and volume confirmation
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donchian_upper[i] and prev_close < donchian_upper[i-1]
            breakout_down = curr_low <= donchian_lower[i] and prev_close > donchian_lower[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
            
            # 12h EMA50 trend filter: rising for long, falling for short
            ema_rising = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            ema_falling = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
            
            if breakout_up and volume_confirm and ema_rising:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0