#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 1d ADX Trend Filter
# Hypothesis: Elder Ray power (bull/bear) confirms trend strength when combined with daily ADX > 20.
# Long when Bull Power > 0 and ADX > 20; Short when Bear Power > 0 and ADX > 20.
# Works in both bull and bear markets by filtering weak trends.
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
    
    # Get daily data for EMA and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA(13) for Elder Ray
    close_daily = df_daily['close'].values
    ema13_daily = pd.Series(close_daily).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Daily ADX(14)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_daily - np.roll(high_daily, 1)
    down_move = np.roll(low_daily, 1) - low_daily
    up_move[0] = down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def smooth(w, period):
        result = np.full_like(w, np.nan)
        if len(w) < period:
            return result
        # Initial smoothed value (simple average)
        result[period-1] = np.nansum(w[:period]) / period
        # Wilder smoothing
        for i in range(period, len(w)):
            if not np.isnan(result[i-1]) and not np.isnan(w[i]):
                result[i] = (result[i-1] * (period-1) + w[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Elder Ray components
    bull_power = high_daily - ema13_daily
    bear_power = ema13_daily - low_daily
    
    # Align to 6h
    ema13_6h = align_htf_to_ltf(prices, df_daily, ema13_daily)
    adx_6h = align_htf_to_ltf(prices, df_daily, adx)
    bull_power_6h = align_htf_to_ltf(prices, df_daily, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_daily, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema13_6h[i]) or np.isnan(adx_6h[i]) or 
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 or ADX < 20
            if bull_power_6h[i] <= 0 or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 or ADX < 20
            if bear_power_6h[i] <= 0 or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong trend: ADX > 20
            if adx_6h[i] > 20:
                if bull_power_6h[i] > 0:  # Uptrend
                    position = 1
                    signals[i] = 0.25
                elif bear_power_6h[i] > 0:  # Downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals