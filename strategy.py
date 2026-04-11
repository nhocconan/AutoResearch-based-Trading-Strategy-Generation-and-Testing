#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_donchian_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return signals
    
    # Daily close for weekly pivot calculation (using previous day's close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from previous week's OHLC
    # Using Monday's open, week's high, week's low, Friday's close
    # For simplicity, use previous week's data
    week_open = df_1w['open'].values
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    week_close = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (week_high + week_low + week_close) / 3
    # Support and resistance levels
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    r3 = week_high + 2 * (pivot - week_low)
    s3 = week_low - 2 * (week_high - pivot)
    
    # Align weekly levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(ema_50_1d_6h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_1d_6h[i]
        
        # Breakout conditions
        breakout_long = price_high > donch_high
        breakout_short = price_low < donch_low
        
        # Weekly pivot direction filter
        above_pivot = price_close > pivot_6h[i]
        below_pivot = price_close < pivot_6h[i]
        
        # Volume confirmation
        volume_ok = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend filter: align with daily EMA50
        trend_up = price_close > ema_trend
        trend_down = price_close < ema_trend
        
        # Entry signals
        long_entry = False
        short_entry = False
        
        # Long: Donchian breakout + above weekly pivot + volume + up trend
        if breakout_long and above_pivot and volume_ok and trend_up:
            long_entry = True
        
        # Short: Donchian breakdown + below weekly pivot + volume + down trend
        if breakout_short and below_pivot and volume_ok and trend_down:
            short_entry = True
        
        # Exit conditions: opposite Donchian level or pivot cross
        exit_long = price_low < donch_low or price_close < pivot_6h[i]
        exit_short = price_high > donch_high or price_close > pivot_6h[i]
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s Donchian breakout filtered by weekly pivot direction and daily trend.
# Enters long when price breaks above 6h Donchian high (20-period) while above weekly pivot,
# with volume confirmation and aligned with daily EMA50 uptrend.
# Enters short when price breaks below 6h Donchian low while below weekly pivot,
# with volume confirmation and aligned with daily EMA50 downtrend.
# Weekly pivot provides institutional reference points; Donchian captures breakouts.
# Daily EMA50 ensures trades align with higher timeframe trend.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
# Target: 15-25 trades per year to minimize fee drag while capturing strong moves.
# Weekly pivot adds institutional intelligence not present in pure Donchian strategies.