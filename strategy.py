#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Breakout with Volume and ADX Filter
# Hypothesis: Daily pivot levels are key support/resistance levels. Price breaking above R1
# with strong volume and trending conditions (ADX > 25) indicates momentum continuation.
# Works in bull markets: breaks above R1 continue upward. Works in bear markets: breaks
# below S1 continue downward. Volume ensures institutional participation, ADX filters
# choppy markets. Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_pivot_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily pivot points
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    daily_r2 = daily_pivot + (prev_daily_high - prev_daily_low)
    daily_s2 = daily_pivot - (prev_daily_high - prev_daily_low)
    
    # Align to 12h timeframe (use previous day's levels)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_daily, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_daily, daily_s2)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: trend strength > 25
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(daily_pivot_aligned[i]) or np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or np.isnan(daily_r2_aligned[i]) or 
            np.isnan(daily_s2_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to daily pivot or volume drops or ADX weakens
            if (close[i] <= daily_pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to daily pivot or volume drops or ADX weakens
            if (close[i] >= daily_pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above daily R1 with volume and strong trend
            if ((high[i] > daily_r1_aligned[i] or high[i] > daily_r2_aligned[i]) and 
                (close[i] > daily_r1_aligned[i] or close[i] > daily_r2_aligned[i]) and 
                vol_filter[i] and adx_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below daily S1 with volume and strong trend
            elif ((low[i] < daily_s1_aligned[i] or low[i] < daily_s2_aligned[i]) and 
                  (close[i] < daily_s1_aligned[i] or close[i] < daily_s2_aligned[i]) and 
                  vol_filter[i] and adx_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals