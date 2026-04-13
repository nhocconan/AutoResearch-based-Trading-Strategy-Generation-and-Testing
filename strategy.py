#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1w ADX trend filter + volume confirmation.
    # Donchian breakouts capture strong momentum moves. ADX > 25 on weekly ensures we only trade in strong trends.
    # Volume confirmation ensures breakout has participation. In ranging markets (ADX < 20), we fade Donchian touches.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 50):
        return np.zeros(n)
    
    # Calculate 1w ADX for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, (dm_plus_smooth / atr_1w) * 100, 0)
    di_minus = np.where(atr_1w > 0, (dm_minus_smooth / atr_1w) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align HTF ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 12h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1w_aligned[i] > 25
        is_ranging = adx_1w_aligned[i] < 20
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if is_trending and volume_filter:
            # In trending market: breakout direction
            long_entry = close[i] > upper_20[i-1]  # Break above upper band
            short_entry = close[i] < lower_20[i-1]  # Break below lower band
        elif is_ranging and volume_filter:
            # In ranging market: fade Donchian touches
            long_entry = close[i] < lower_20[i-1] and close[i] > lower_20[i-1] * 0.998  # Near lower band
            short_entry = close[i] > upper_20[i-1] and close[i] < upper_20[i-1] * 1.002  # Near upper band
        
        # Exit conditions: opposite Donchian touch or regime change to ranging
        long_exit = False
        short_exit = False
        
        if is_trending:
            # Exit when price touches opposite band
            long_exit = close[i] < lower_20[i-1]
            short_exit = close[i] > upper_20[i-1]
        else:  # ranging
            # Exit when price moves back toward center
            long_exit = close[i] > (upper_20[i-1] + lower_20[i-1]) / 2
            short_exit = close[i] < (upper_20[i-1] + lower_20[i-1]) / 2
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_breakout_adx_regime_v1"
timeframe = "12h"
leverage = 1.0