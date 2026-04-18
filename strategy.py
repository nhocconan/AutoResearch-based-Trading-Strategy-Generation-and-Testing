#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume spike and 1-day ADX trend filter.
# Camarilla levels (H3/L3) act as reversal points in ranging markets.
# Volume spike confirms institutional interest at these levels.
# Daily ADX > 25 ensures we only trade in trending conditions to avoid false reversals in chop.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (buy at L3 in uptrend) and bear markets (sell at H3 in downtrend).
name = "4h_Camarilla_H3L3_1dADX_Volume_Spike"
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
    
    # Get daily data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for previous day (using 1-day OHLC)
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    # We need previous day's OHLC, so we shift by 1
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla H3 and L3 for previous day
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 6
    camarilla_l3 = close_1d - 1.1 * range_1d / 6
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate daily ADX (14-period) for trend strength
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_smooth = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    period = 14
    if len(tr) >= period:
        tr_smooth[period-1] = np.nanmean(tr[:period])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[:period])
        for i in range(period, len(tr)):
            if not np.isnan(tr_smooth[i-1]) and not np.isnan(tr[i]):
                tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/period) + tr[i]
                dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/period) + dm_plus[i]
                dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/period) + dm_minus[i]
            else:
                tr_smooth[i] = np.nan
                dm_plus_smooth[i] = np.nan
                dm_minus_smooth[i] = np.nan
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(len(tr)):
        if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
            if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        adx[period-1] = np.nanmean(dx[:period])
        for i in range(period, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            else:
                adx[i] = np.nan
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price at L3 level with volume spike in uptrend
            long_condition = (close[i] <= camarilla_l3_aligned[i] * 1.002 and  # Allow small buffer
                             close[i] >= camarilla_l3_aligned[i] * 0.998 and
                             vol_spike and trending)
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: price at H3 level with volume spike in downtrend
            elif (close[i] >= camarilla_h3_aligned[i] * 0.998 and  # Allow small buffer
                  close[i] <= camarilla_h3_aligned[i] * 1.002 and
                  vol_spike and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price moves back above L3 or ADX drops below 20 (trend weakening)
            exit_condition = (close[i] > camarilla_l3_aligned[i] * 1.01 or 
                             adx_aligned[i] < 20)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves back below H3 or ADX drops below 20 (trend weakening)
            exit_condition = (close[i] < camarilla_h3_aligned[i] * 0.99 or 
                             adx_aligned[i] < 20)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals