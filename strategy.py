#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d Trend Filter and Volume Confirmation
# Uses Williams %R (14) to identify overbought/oversold conditions.
# Long when %R < -80 (oversold) with 1d ADX > 25 (uptrend) and volume > 1.5x average.
# Short when %R > -20 (overbought) with 1d ADX > 25 (downtrend) and volume > 1.5x average.
# Exit when %R returns to -50 (mean reversion) or volatility expands.
# Works in both bull and bear markets by only taking counter-trend entries in the direction of the 1d trend.
# Target: 20-40 trades/year to minimize fee decay while capturing mean reversion moves in trending markets.
# Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smooth(tr, period_adx)
    dm_plus_smooth = wilders_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilders_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.full(n_1d, np.nan)
    di_minus = np.full(n_1d, np.nan)
    dx = np.full(n_1d, np.nan)
    
    for i in range(len(tr_smooth)):
        if np.isnan(tr_smooth[i]) or tr_smooth[i] == 0:
            di_plus[i] = 0
            di_minus[i] = 0
        else:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
    
    for i in range(len(dx)):
        if np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or (di_plus[i] + di_minus[i]) == 0:
            dx[i] = 0
        else:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX = Wilder's smoothed DX
    adx_1d = wilders_smooth(dx, period_adx)
    
    # Williams %R on 4h (14-period)
    williams_period = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(williams_period, n):
        highest_high[i] = np.max(high[i-williams_period:i])
        lowest_low[i] = np.min(low[i-williams_period:i])
        if highest_high[i] - lowest_low[i] != 0:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(williams_period, vol_period) + period_adx
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Williams %R: < -80 oversold (long), > -20 overbought (short)
        # 2. ADX > 25: trending regime on 1d
        # 3. Volume confirmation: > 1.5x average volume
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        trend_up = adx_1d_aligned[i] > 25
        trend_down = adx_1d_aligned[i] > 25  # ADX > 25 indicates trend, direction from price
        volume_confirmation = vol_ratio > 1.5
        mean_reversion = -50 <= williams_r[i] <= -50  # exit at midpoint
        
        if position == 0:
            # Long: oversold during uptrend with volume
            if oversold and trend_up and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: overbought during downtrend with volume
            elif overbought and trend_down and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: %R returns to -50 (mean reversion) or volatility expands
            if williams_r[i] >= -50:  # exited at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: %R returns to -50 (mean reversion) or volatility expands
            if williams_r[i] <= -50:  # exited at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0