#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot R1/S1 breakout with weekly ADX trend filter and volume confirmation.
# Camarilla levels from prior day provide precise intraday support/resistance.
# Weekly ADX > 25 ensures we only trade in trending conditions to avoid chop.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1).
name = "6h_Camarilla_R1_S1_Breakout_WeeklyADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get weekly data for ADX trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_w[1:] - high_w[:-1]
    down_move = low_w[:-1] - low_w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    adx_period = 14
    tr_smooth = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(tr, np.nan)
    minus_dm_smooth = np.full_like(tr, np.nan)
    
    if len(tr) >= adx_period:
        tr_smooth[adx_period-1] = np.nanmean(tr[:adx_period])
        plus_dm_smooth[adx_period-1] = np.nanmean(plus_dm[:adx_period])
        minus_dm_smooth[adx_period-1] = np.nanmean(minus_dm[:adx_period])
        
        for i in range(adx_period, len(tr)):
            if not np.isnan(tr_smooth[i-1]) and not np.isnan(tr[i]):
                tr_smooth[i] = tr_smooth[i-1] * (1 - 1/adx_period) + tr[i] * (1/adx_period)
            else:
                tr_smooth[i] = np.nan
                
            if not np.isnan(plus_dm_smooth[i-1]) and not np.isnan(plus_dm[i]):
                plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - 1/adx_period) + plus_dm[i] * (1/adx_period)
            else:
                plus_dm_smooth[i] = np.nan
                
            if not np.isnan(minus_dm_smooth[i-1]) and not np.isnan(minus_dm[i]):
                minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - 1/adx_period) + minus_dm[i] * (1/adx_period)
            else:
                minus_dm_smooth[i] = np.nan
    
    # Calculate DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.nanmean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = adx[i-1] * (1 - 1/adx_period) + dx[i] * (1/adx_period)
            else:
                adx[i] = np.nan
    
    # Align weekly ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 24-period average volume for confirmation (4 days worth)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND trend filter
            long_breakout = close[i] > camarilla_r1_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < camarilla_s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < camarilla_s1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > camarilla_r1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals