#!/usr/bin/env python3
"""
Hypothesis: 1-day mean-reversion at weekly Bollinger Bands (20,2) with volume confirmation and weekly ADX filter.
Enters long when price touches lower BB and reverses up with volume spike in low-volatility regime (ADX < 20).
Enters short when price touches upper BB and reverses down with volume spike in low-volatility regime.
Uses daily close for entry and exit at opposite band. Targets 10-25 trades per year to minimize fee drag.
Works in both bull (buy dips) and bear (sell rallies) by fading extremes in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Bollinger Bands and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Bollinger Bands (2 standard deviations)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume spike: volume > 1.8x 20-period average (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr0 = np.max([high_1w[0] - low_1w[0], 
                  np.abs(high_1w[0] - close_1w[0]), 
                  np.abs(low_1w[0] - close_1w[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values with Wilder's smoothing (using EMA-like approach)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = WilderSmoothing(tr, 14)
    dm_plus_14 = WilderSmoothing(dm_plus, 14)
    dm_minus_14 = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmoothing(dx, 14)
    
    # Low trend filter: ADX < 20 (strong ranging market)
    low_trend = adx < 20
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    low_trend_aligned = align_htf_to_ltf(prices, df_1w, low_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: BB touch + volume spike + low trend
        touch_upper = close[i] >= bb_upper_aligned[i] * 0.999  # Allow small slippage
        touch_lower = close[i] <= bb_lower_aligned[i] * 1.001  # Allow small slippage
        
        # Require reversal confirmation: close in opposite direction from touch
        reversal_up = close[i] > open_prices[i] if 'open' in prices.columns else close[i] > close[i-1]
        reversal_down = close[i] < open_prices[i] if 'open' in prices.columns else close[i] < close[i-1]
        
        vol_confirm = vol_spike_aligned[i] > 0.5
        trend_filter = low_trend_aligned[i] > 0.5
        
        long_entry = touch_lower and reversal_up and vol_confirm and trend_filter
        short_entry = touch_upper and reversal_down and vol_confirm and trend_filter
        
        # Exit at opposite Bollinger Band
        exit_long = position == 1 and close[i] >= bb_upper_aligned[i]
        exit_short = position == -1 and close[i] <= bb_lower_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_bb_reversion_volume_lowtrend"
timeframe = "1d"
leverage = 1.0