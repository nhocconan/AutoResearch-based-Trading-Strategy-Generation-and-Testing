#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly ADX trend filter and volume confirmation.
# Weekly Donchian channels provide strong trend-following signals based on weekly extremes.
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
name = "1d_WeeklyDonchian_ADX_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and ADX (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period) using previous week's data to avoid look-ahead
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band = high_20w
    lower_band = low_20w
    
    # Calculate weekly ADX (14-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    def wilders_smooth(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) >= period:
            smoothed[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(smoothed[i-1]) and not np.isnan(data[i]):
                    smoothed[i] = smoothed[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    smoothed[i] = np.nan
        return smoothed
    
    atr_w = wilders_smooth(tr_w, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_w != 0, 100 * dm_plus_smooth / atr_w, 0)
    di_minus = np.where(atr_w != 0, 100 * dm_minus_smooth / atr_w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align weekly indicators to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations (20+14+14+14 for ADX)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper band AND trending AND volume confirmation
            long_breakout = close[i] > upper_band_aligned[i]
            if trending and vol_confirm and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND trending AND volume confirmation
            elif trending and vol_confirm and close[i] < lower_band_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < lower_band_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > upper_band_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals