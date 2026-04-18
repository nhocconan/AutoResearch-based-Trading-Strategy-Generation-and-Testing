#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 20-period Donchian breakout with daily volume confirmation and daily ADX trend filter.
# Donchian channels provide clear breakout levels based on recent price extremes.
# Daily volume confirmation ensures breakouts are supported by increased participation.
# Daily ADX > 25 filters for trending markets, avoiding false breakouts in ranges.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above upper band with rising ADX) and bear markets 
# (breakdowns below lower band with rising ADX).
name = "4h_Donchian20_DailyVolume_ADX"
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
    
    # Get daily data for volume and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian bands using previous day's data to avoid look-ahead
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high = high_20
    donchian_low = low_20
    
    # Calculate daily ADX for trend strength
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                       np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                        np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
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
    
    # Calculate 20-period average daily volume for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian bands, ADX, and volume MA to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume above average
        vol_confirm = df_1d['volume'].iloc[-1] > vol_ma_20_aligned[i] if len(df_1d) > 0 else False
        # For simplicity, use the aligned volume MA directly - current volume > MA
        vol_confirm = volume[i] > vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume confirmation AND ADX > 25
            long_breakout = close[i] > donchian_high_aligned[i]
            strong_trend = adx_aligned[i] > 25
            
            if vol_confirm and long_breakout and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band AND volume confirmation AND ADX > 25
            elif (vol_confirm and 
                  close[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower Donchian band
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Donchian band
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals