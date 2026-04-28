#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend with 1d ADX regime filter and volume confirmation
# Supertrend captures trend direction with built-in ATR stop mechanism
# 1d ADX > 25 filters for trending markets (avoids choppy ranging periods)
# Volume confirmation > 2.0x 20-bar average ensures institutional participation
# Long when Supertrend = uptrend, 1d ADX > 25, volume spike
# Short when Supertrend = downtrend, 1d ADX > 25, volume spike
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn
# Targets 20-50 trades/year to avoid fee drag while capturing major trends

name = "4h_Supertrend_1dADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, (plus_dm_smooth / atr_1d) * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, (minus_dm_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100, 0)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Supertrend
    atr_period = 10
    multiplier = 3.0
    
    # True Range for 4h
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum(np.maximum(tr1_4h, tr2_4h), tr3_4h)
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    # ATR using Wilder's smoothing
    atr_4h = wilder_smooth(tr_4h, atr_period)
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr_4h)
    lower_band = hl2 - (multiplier * atr_4h)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    trend = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_atr = atr_period
    if start_atr < n:
        supertrend[start_atr] = upper_band[start_atr]
        trend[start_atr] = 1
    
    # Calculate Supertrend iteratively
    for i in range(start_atr + 1, n):
        if np.isnan(atr_4h[i]) or np.isnan(close[i-1]):
            supertrend[i] = supertrend[i-1]
            trend[i] = trend[i-1]
            continue
            
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(trend[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        adx_val = adx_1d_aligned[i]
        st_value = supertrend[i]
        st_trend = trend[i]
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        is_trending = adx_val > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Supertrend uptrend, trending market, volume spike
            if st_trend == 1 and is_trending and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Supertrend downtrend, trending market, volume spike
            elif st_trend == -1 and is_trending and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on trend change
            # Exit when Supertrend turns downtrend
            if st_trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on trend change
            # Exit when Supertrend turns uptrend
            if st_trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals