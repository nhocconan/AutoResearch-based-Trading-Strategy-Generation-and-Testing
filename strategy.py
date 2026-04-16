#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND 12h ADX > 25 (trending) AND volume > 1.5x average volume.
# Short when price breaks below Donchian lower channel AND 12h ADX > 25 (trending) AND volume > 1.5x average volume.
# Exit when price touches Donchian middle (20-period average of high/low) OR ADX < 20 (range regime).
# Uses discrete position size 0.25. Donchian provides clear structure, ADX filters choppy markets, volume confirms breakout strength.
# Targets 20-50 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 12h data once for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    # Upper channel = max(high, 20)
    # Lower channel = min(low, 20)
    # Middle channel = (upper + lower) / 2
    from pandas import Series
    high_series_4h = Series(high_4h)
    low_series_4h = Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # === 12h Indicators: ADX (14-period) for trend strength ===
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    # +DM = max(high - high_prev, 0) if high - high_prev > low_prev - low else 0
    # -DM = max(low_prev - low, 0) if low_prev - low > high - high_prev else 0
    # smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    # +DI = 100 * smoothed +DM / smoothed TR
    # -DI = 100 * smoothed -DM / smoothed TR
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX (Wilder's smoothing)
    
    high_12h_series = Series(high_12h)
    low_12h_series = Series(low_12h)
    close_12h_series = Series(close_12h)
    
    # True Range
    tr1 = high_12h_series - low_12h_series
    tr2 = abs(high_12h_series - close_12h_series.shift(1))
    tr3 = abs(low_12h_series - close_12h_series.shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # +DM and -DM
    up_move = high_12h_series - high_12h_series.shift(1)
    down_move = low_12h_series.shift(1) - low_12h_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # first value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period_adx = 14
    tr_smoothed = wilder_smoothing(tr_12h, period_adx)
    plus_dm_smoothed = wilder_smoothing(plus_dm, period_adx)
    minus_dm_smoothed = wilder_smoothing(minus_dm, period_adx)
    
    # Avoid division by zero
    plus_di_12h = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di_12h = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      100 * abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 0)
    adx_12h = wilder_smoothing(dx_12h, period_adx)
    
    # Volume average (20-period)
    volume_ma_20 = Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)  # volume MA is already 4h
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian(20) + ADX(14) + Wilder smoothing needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        adx = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price touches middle channel OR ADX < 20 (range regime)
            if (price <= middle) or (adx < 20):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price touches middle channel OR ADX < 20 (range regime)
            if (price >= middle) or (adx < 20):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = vol > (1.5 * vol_ma)
            
            # LONG: Price breaks above upper channel AND ADX > 25 (strong trend) AND volume confirmed
            if (price > upper) and (adx > 25) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower channel AND ADX > 25 (strong trend) AND volume confirmed
            elif (price < lower) and (adx > 25) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_ADX12h_VolumeConfirmation_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0