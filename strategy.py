#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Pivot R1/S1 breakout on daily timeframe with volume confirmation
# and ADX-based trend filter. Works in bull/bear by capturing breakouts from weekly
# pivot levels (strong support/resistance) with volume confirmation and trend filter
# to avoid whipsaws. Target: 15-25 trades/year.
name = "1d_WeeklyPivot_R1S1_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (using previous week)
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_close = np.roll(close_1w, 1)
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Weekly Pivot = (H + L + C) / 3
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly R1 = C + (H - L) * 1.1 / 12
    weekly_r1 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 12.0
    # Weekly S1 = C - (H - L) * 1.1 / 12
    weekly_s1 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 12.0
    
    # Align weekly levels to daily timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_1d = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # ADX filter: only trade when ADX > 25 (trending market)
    # Calculate ADX components
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, plus_dm_smooth / tr_smooth * 100, 0)
    minus_di = np.where(tr_smooth != 0, minus_dm_smooth / tr_smooth * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Time filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        trending_market = adx[i] > 25
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and trend
            if price > r1_1d[i] and volume_confirmed and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and trend
            elif price < s1_1d[i] and volume_confirmed and trending_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly pivot (mean reversion)
            if price < pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly pivot (mean reversion)
            if price > pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals