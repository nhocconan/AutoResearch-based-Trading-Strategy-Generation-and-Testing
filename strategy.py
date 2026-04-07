#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + ADX Trend + Volume Confirmation
# Hypothesis: Combines Elder Ray (bull/bear power) with ADX trend strength and volume
# to capture momentum in both bull and bear markets. Elder Ray > 0 indicates bullish
# pressure, < 0 bearish pressure. ADX > 25 filters for trending conditions.
# Works in bull via bull power + ADX, in bear via bear power + ADX, and avoids whipsaws
# in ranging markets. Target: 15-30 trades/year to minimize fee drag.
name = "6h_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA(13) - used in Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) from daily close
    daily_close = df_1d['close'].values
    daily_ema13 = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_6h = align_htf_to_ltf(prices, df_1d, daily_ema13)
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Get weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) from weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(weekly_high - weekly_low)
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low), 
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)), 
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # Initial average
        result[period-1] = np.nansum(data[:period])
        # Wilder smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend strength filter
        trending = adx_6h[i] > 25
        
        if position == 1:  # Long position
            # Exit: Bear power becomes positive (momentum fading) or trend weakens
            if bear_power[i] > 0 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bull power becomes negative (momentum fading) or trend weakens
            if bull_power[i] < 0 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Bull power > 0 with volume and trending ADX
            if bull_power[i] > 0 and vol_confirm and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear power < 0 with volume and trending ADX
            elif bear_power[i] < 0 and vol_confirm and trending:
                position = -1
                signals[i] = -0.25
    
    return signals