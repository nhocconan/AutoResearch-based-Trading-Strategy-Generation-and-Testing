#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Breakout with Volume and ADX Filter
# Hypothesis: Daily pivot levels are key support/resistance. Price breaking above R1 or below S1 with volume and trend strength (ADX>25) indicates momentum continuation. Works in bull/bear: breaks continue in trend direction. Volume ensures institutional participation. ADX filters choppy markets.
# Target: 12-37 trades/year (50-150 over 4 years).

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
    
    # ADX filter: ADX > 25 indicates trending market
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[1] if len(tr) > 1 else 0  # Fix first value
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(tr)
    minus_dm_sum = np.zeros_like(tr)
    
    # Initialize first values
    tr_sum[0] = tr[0]
    plus_dm_sum[0] = plus_dm[0]
    minus_dm_sum[0] = minus_dm[0]
    
    # Wilder smoothing
    for i in range(1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / atr_period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / atr_period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / atr_period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[atr_period-1] = dx[atr_period-1]  # First ADX value
    for i in range(atr_period, len(dx)):
        adx[i] = (adx[i-1] * (atr_period-1) + dx[i]) / atr_period
    
    # Set ADX values before period to 0
    adx[:atr_period-1] = 0
    
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
            # Exit: price falls to daily pivot or volume/ADX drops
            if (close[i] <= daily_pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to daily pivot or volume/ADX drops
            if (close[i] >= daily_pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above daily R1 with volume and trend
            if ((high[i] > daily_r1_aligned[i] or high[i] > daily_r2_aligned[i]) and 
                (close[i] > daily_r1_aligned[i] or close[i] > daily_r2_aligned[i]) and 
                vol_filter[i] and adx_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below daily S1 with volume and trend
            elif ((low[i] < daily_s1_aligned[i] or low[i] < daily_s2_aligned[i]) and 
                  (close[i] < daily_s1_aligned[i] or close[i] < daily_s2_aligned[i]) and 
                  vol_filter[i] and adx_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals