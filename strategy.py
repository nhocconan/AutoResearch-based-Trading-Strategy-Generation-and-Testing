#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d pivot levels (R1/S1) + volume confirmation + ADX trend filter
# Long when price breaks above R1 with volume and ADX > 25 (trending up)
# Short when price breaks below S1 with volume and ADX > 25 (trending down)
# Exit when price returns to pivot or ADX < 20 (trend weakening)
# Designed to work in both bull (trend following) and bear (trend following short) markets
# Target: 20-40 trades/year to avoid fee drag

name = "4h_1d_Pivot_R1S1_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*Pivot - Low
    r1 = 2 * pivot - low_1d
    # S1 = 2*Pivot - High
    s1 = 2 * pivot - high_1d
    
    # Align daily pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period) on 4h data
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DM14 = smoothed +DM, -DM14 = smoothed -DM, TR14 = smoothed TR
    # +DI14 = 100 * +DM14 / TR14, -DI14 = 100 * -DM14 / TR14
    # DX = 100 * abs(+DI14 - -DI14) / (+DI14 + -DI14)
    # ADX = smoothed DX
    
    # Calculate directional movement
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    plus_dm14 = wilders_smoothing(plus_dm, period)
    minus_dm14 = wilders_smoothing(minus_dm, period)
    tr14 = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smoothing(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough for indicators to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        adx_trending = adx[i] > 25
        adx_not_weak = adx[i] > 20  # For exit condition
        
        if position == 0:
            # Long: Price breaks above R1 with volume and ADX trending up
            if price > r1_4h[i] and volume_confirmed and adx_trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and ADX trending up
            elif price < s1_4h[i] and volume_confirmed and adx_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot OR ADX weakens (trend ending)
            if price < pivot_4h[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot OR ADX weakens (trend ending)
            if price > pivot_4h[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals