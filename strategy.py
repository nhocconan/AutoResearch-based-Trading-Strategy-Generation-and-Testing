#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day/1-week Donchian breakout + volume confirmation + ADX trend filter.
# Uses weekly Donchian channels for trend direction and daily levels for entry.
# Long when price breaks above daily upper band in weekly uptrend with volume confirmation.
# Short when price breaks below daily lower band in weekly downtrend with volume confirmation.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Designed for 15-25 trades/year with focus on quality over quantity.

name = "12h_1d1w_donchian_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period) for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper and lower bands for weekly Donchian (20-period)
    upper_20_1w = np.full_like(high_1w, np.nan)
    lower_20_1w = np.full_like(low_1w, np.nan)
    
    for i in range(19, len(high_1w)):
        upper_20_1w[i] = np.max(high_1w[i-19:i+1])
        lower_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly trend: price above upper band = uptrend, below lower band = downtrend
    weekly_uptrend = high_1w > upper_20_1w
    weekly_downtrend = low_1w < lower_20_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Calculate daily Donchian channels (20-period) for entry signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    upper_20_1d = np.full_like(high_1d, np.nan)
    lower_20_1d = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        upper_20_1d[i] = np.max(high_1d[i-19:i+1])
        lower_20_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Daily average volume (20-period) for volume filter
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily levels and volume to 12h
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate ADX (14-period) on daily data for trend strength filter
    # Need True Range, +DM, -DM first
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / atr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = wilder_smooth(dx_14, 14)
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_1d_aligned[i]) or np.isnan(lower_20_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.8 * daily average volume
        vol_filter = volume[i] > 1.8 * vol_avg_aligned[i]
        
        # ADX filter: only trade when ADX > 25 (trending market)
        adx_filter = adx_aligned[i] > 25
        
        # Determine weekly trend direction
        is_weekly_uptrend = weekly_uptrend_aligned[i]
        is_weekly_downtrend = weekly_downtrend_aligned[i]
        
        # Entry conditions
        breakout_long = (high[i] >= upper_20_1d_aligned[i] and vol_filter and adx_filter and is_weekly_uptrend)
        breakout_short = (low[i] <= lower_20_1d_aligned[i] and vol_filter and adx_filter and is_weekly_downtrend)
        
        # Exit when price returns to the opposite Donchian band or ADX weakens
        exit_long = (position == 1 and 
                    (low[i] <= lower_20_1d_aligned[i] or adx_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (high[i] >= upper_20_1d_aligned[i] or adx_aligned[i] < 20))
        
        # Priority: breakout > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals