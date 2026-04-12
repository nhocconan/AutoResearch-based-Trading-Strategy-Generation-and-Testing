#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_pivot_reversion_v1
# Uses weekly pivot points (from prior week) as mean reversion levels on 6h chart.
# Long when price touches weekly S1 with bullish rejection (close > open) and volume confirmation.
# Short when price touches weekly R1 with bearish rejection (close < open) and volume confirmation.
# Exits when price crosses weekly pivot point (mean reversion).
# Only trades when 1d ADX < 25 (ranging market) to avoid trending whipsaws.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in ranging markets via pivot reversion and avoids trending markets via ADX filter.
# Focus on BTC/ETH as primary targets.

name = "6h_1d_1w_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 5 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (from prior week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot (using prior week's OHLC)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation: volume > 1.3 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # ADX filter: only trade when ADX < 25 (ranging market)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Wilder's smoothing
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            alpha = 1.0 / period
            result[period-1] = np.nansum(data[:period]) / period
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Avoid division by zero
        dx = np.zeros_like(atr)
        mask = (dm_plus_smooth + dm_minus_smooth) != 0
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_filter = adx < 25  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Check if filters pass
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price touches S1 with bullish rejection
        if (low[i] <= s1_1w_aligned[i] * 1.001 and  # allow 0.1% tolerance
            close[i] > open_price[i] and  # bullish candle
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price touches R1 with bearish rejection
        elif (high[i] >= r1_1w_aligned[i] * 0.999 and  # allow 0.1% tolerance
              close[i] < open_price[i] and  # bearish candle
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses weekly pivot (mean reversion)
        elif position == 1 and close[i] >= pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals