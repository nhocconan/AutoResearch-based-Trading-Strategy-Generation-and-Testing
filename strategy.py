#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot with Volume and ADX Filter
# Hypothesis: Weekly pivot points (based on weekly high/low/close) act as strong support/resistance on daily timeframe.
# Price bouncing off weekly pivot with volume confirmation and ADX trend filter captures reversals in both bull and bear markets.
# Weekly pivot provides structure, volume confirms institutional interest, ADX filters choppy markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_weekly_pivot_volume_adx_v1"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to daily
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # ADX(14) on daily for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First TR
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan, dtype=float)
            if len(data) < period:
                return result
            result[period-1] = np.mean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = WilderSmooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price moves below pivot or ADX weakens
            if close[i] < weekly_pivot_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price moves above pivot or ADX weakens
            if close[i] > weekly_pivot_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            # ADX filter: trending market (ADX > 25)
            trend_filter = adx[i] > 25
            
            if vol_confirm and trend_filter:
                # Long setup: price near weekly pivot from below
                if close[i] >= weekly_pivot_aligned[i] * 0.995 and close[i] <= weekly_pivot_aligned[i] * 1.005:
                    # Determine direction based on price action relative to pivot
                    if i > 0 and close[i-1] < weekly_pivot_aligned[i-1] and close[i] >= weekly_pivot_aligned[i]:
                        # Crossed above pivot - go long
                        position = 1
                        signals[i] = 0.25
                    elif i > 0 and close[i-1] > weekly_pivot_aligned[i-1] and close[i] <= weekly_pivot_aligned[i]:
                        # Crossed below pivot - go short
                        position = -1
                        signals[i] = -0.25
    
    return signals