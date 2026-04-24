#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ADX trend filter (ADX > 25 indicates strong trend).
- Entry: Long when price breaks above Donchian upper AND ADX > 25 AND volume > 1.5 * avg_volume;
         Short when price breaks below Donchian lower AND ADX > 25 AND volume > 1.5 * avg_volume.
- Exit: Opposite Donchian breakout OR ADX falls below 20 (trend weakening).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear structure; ADX filters for trending markets only; volume confirms momentum.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with trend/volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nanmean(data[i-period+1:i+1])
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = WilderSmoothing(tr, period)
    dm_plus_smooth = WilderSmoothing(dm_plus, period)
    dm_minus_smooth = WilderSmoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmoothing(dx, period)
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    weak_trend = adx < 20  # for exit
    
    # Align 1d indicators to 4h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend, additional_delay_bars=1)
    weak_trend_aligned = align_htf_to_ltf(prices, df_1d, weak_trend, additional_delay_bars=1)
    
    # Calculate Donchian channels on 4h data (20-period)
    donchian_window = 20
    # Upper band: highest high over past donchian_window periods
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    # Lower band: lowest low over past donchian_window periods
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Breakout signals
    breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    
    # Handle first element for roll
    breakout_up[0] = False
    breakout_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_window)  # Need sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(strong_trend_aligned[i]) or np.isnan(weak_trend_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite breakout OR trend weakening (ADX < 20)
        if position != 0:
            # Exit long: breakout down OR trend weakening
            if position == 1:
                if breakout_down[i] or weak_trend_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: breakout up OR trend weakening
            elif position == -1:
                if breakout_up[i] or weak_trend_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        if position == 0:
            # Long: breakout up AND strong trend AND volume spike
            if breakout_up[i] and strong_trend_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down AND strong trend AND volume spike
            elif breakout_down[i] and strong_trend_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADXTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0