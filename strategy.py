#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for ADX(14) regime filter (ADX > 25 = trending market).
- Entry: Long when price breaks above Donchian upper channel (20-period high) AND 12h ADX > 25 AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower channel (20-period low) AND 12h ADX > 25 AND volume > 1.5 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or ADX < 20 (regime change to ranging).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear structural breakouts; ADX filter ensures we only trade in trending regimes (avoiding whipsaws in ranges); volume confirmation avoids false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with ADX filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ , DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Get 4h data for Donchian channels and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: ADX < 20 (regime change to ranging) or opposite signal
        if position != 0:
            if adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            elif position == 1 and curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Donchian breakout
        bullish_breakout = curr_high > donchian_high_aligned[i]  # Break above upper channel
        bearish_breakout = curr_low < donchian_low_aligned[i]    # Break below lower channel
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Volume confirmation (1.5x average volume)
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and trending:
                # Long: Price breaks above Donchian upper channel AND trending
                if bullish_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian lower channel AND trending
                elif bearish_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX14_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0