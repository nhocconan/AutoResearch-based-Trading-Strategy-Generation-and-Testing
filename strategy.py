#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 1d ADX Trend Filter
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h with 1d ADX > 25 to confirm trend strength.
# Long when Bull Power > 0 and Bear Power < 0 in strong trend; Short when Bear Power > 0 and Bull Power < 0.
# Works in bull/bear by only taking trades in strong trends (ADX > 25), avoiding chop.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_elder_ray_1d_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray (6h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ADX on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    up_move = high_daily - np.roll(high_daily, 1)
    down_move = np.roll(low_daily, 1) - low_daily
    up_move[0] = np.nan
    down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_period = 14
    tr_atr = wilders_smoothing(tr, atr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, atr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, atr_period)
    
    # Avoid division by zero
    plus_di = np.where(tr_atr != 0, 100 * plus_dm_smoothed / tr_atr, 0)
    minus_di = np.where(tr_atr != 0, 100 * minus_dm_smoothed / tr_atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    # Align ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_6h[i] > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens or Bear Power becomes positive
            if not strong_trend or bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: trend weakens or Bull Power becomes positive
            if not strong_trend or bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: Bull Power > 0 and Bear Power < 0 in strong uptrend
            if strong_trend and bull_power[i] > 0 and bear_power[i] < 0:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power > 0 and Bull Power < 0 in strong downtrend
            elif strong_trend and bear_power[i] > 0 and bull_power[i] < 0:
                position = -1
                signals[i] = -0.25
    
    return signals