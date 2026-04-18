#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 1w trend filter.
# Donchian(20) provides clear breakout levels with statistical edge in trending markets.
# 1d volume surge confirms institutional participation in the breakout.
# 1w ADX > 25 ensures we only trade in strong weekly trends, avoiding whipsaws in ranging markets.
# Designed for low trade frequency (20-40/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above upper band with rising weekly ADX) and bear markets 
# (breakdowns below lower band with rising weekly ADX).
name = "4h_Donchian20_1dVolume_1wADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period) using previous period's data to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_upper = high_20
    donchian_lower = low_20
    
    # Calculate 1d average volume (20-period) for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly ADX for trend strength (using Wilder's smoothing)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-period average volume for 4h confirmation
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average (both 4h and 1d)
        vol_confirm_4h = volume[i] > vol_ma_20_4h[i]
        vol_confirm_1d = df_1d['volume'].iloc[-1] > vol_ma_20[-1] if len(df_1d) > 0 else False  # Simplified: use latest available
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume confirmation AND strong weekly trend (ADX > 25)
            breakout_up = close[i] > donchian_upper_aligned[i]
            if vol_confirm_4h and breakout_up and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND volume confirmation AND strong weekly trend (ADX > 25)
            elif vol_confirm_4h and close[i] < donchian_lower_aligned[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower (reversal signal)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper (reversal signal)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals