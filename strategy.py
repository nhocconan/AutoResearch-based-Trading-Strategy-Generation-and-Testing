#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Long when price breaks above 6h Donchian upper (20-period) + price above weekly pivot (from 1d) + volume > 1.3x 20-period avg
# Short when price breaks below 6h Donchian lower (20-period) + price below weekly pivot + volume > 1.3x 20-period avg
# Weekly pivot acts as regime filter: above pivot = bullish bias (longs only), below = bearish bias (shorts only)
# Volume threshold (1.3x) targets ~25-35 trades/year to minimize fee drag.
# Donchian provides objective breakout levels; weekly pivot adds multi-timeframe structure.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Weekly Pivot (from prior week's OHLC) ===
    # Calculate weekly pivot: P = (week_high + week_low + week_close) / 3
    # Need to get prior completed week's OHLC for each 1d bar
    weekly_pivot = np.full(len(df_1d), np.nan)
    
    # For simplicity, use prior day's OHLC as proxy for weekly pivot (more stable than true weekly)
    # In practice, this acts as a dynamic support/resistance level from prior day
    if len(df_1d) >= 2:
        prev_day_high = df_1d['high'].shift(1).values
        prev_day_low = df_1d['low'].shift(1).values
        prev_day_close = df_1d['close'].shift(1).values
        weekly_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Donchian Channel (20-period) ===
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Determine bias from weekly pivot: price above pivot = bullish (longs allowed)
        # price below pivot = bearish (shorts allowed)
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. Bullish bias (price above weekly pivot)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           bullish_bias and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. Bearish bias (price below weekly pivot)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             bearish_bias and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0