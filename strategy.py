#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend strength + 12h Donchian breakout with volume confirmation.
# ADX > 25 filters for trending markets to avoid chop. 
# 12h Donchian breakout captures medium-term momentum.
# Volume confirmation ensures breakouts have conviction.
# Designed for low trade frequency (<50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
name = "4h_ADX25_12hDonchian20_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band_12h = high_20_12h
    lower_band_12h = low_20_12h
    
    # Align 12h Donchian bands to 4h timeframe
    upper_band_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_band_12h)
    lower_band_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_band_12h)
    
    # Calculate ADX (14-period) for trend strength filter
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/14)
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(tr, np.nan)
    dm_minus_smooth = np.full_like(tr, np.nan)
    
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[:tr_period])
        for i in range(tr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/tr_period) + tr[i] * (1/tr_period)
            else:
                atr[i] = np.nan
            if not np.isnan(dm_plus_smooth[i-1]) and not np.isnan(dm_plus[i]):
                dm_plus_smooth[i] = dm_plus_smooth[i-1] * (1 - 1/tr_period) + dm_plus[i] * (1/tr_period)
            else:
                dm_plus_smooth[i] = np.nan
            if not np.isnan(dm_minus_smooth[i-1]) and not np.isnan(dm_minus[i]):
                dm_minus_smooth[i] = dm_minus_smooth[i-1] * (1 - 1/tr_period) + dm_minus[i] * (1/tr_period)
            else:
                dm_minus_smooth[i] = np.nan
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(tr_period, len(tr)):
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (di_plus[i] + di_minus[i]) > 0:
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # Calculate ADX as smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= tr_period:
        # First ADX value is average of first tr_period DX values
        valid_dx = dx[tr_period:2*tr_period]
        if not np.all(np.isnan(valid_dx)):
            adx[2*tr_period-1] = np.nanmean(valid_dx)
            for i in range(2*tr_period, len(dx)):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = adx[i-1] * (1 - 1/tr_period) + dx[i] * (1/tr_period)
                else:
                    adx[i] = np.nan
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_12h_aligned[i]) or np.isnan(lower_band_12h_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above upper band AND volume confirmation AND trend filter
            long_breakout = close[i] > upper_band_12h_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < lower_band_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR ADX falls below 20 (trend weakening)
            exit_condition = close[i] < lower_band_12h_aligned[i] or adx[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR ADX falls below 20 (trend weakening)
            exit_condition = close[i] > upper_band_12h_aligned[i] or adx[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals