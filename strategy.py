#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width squeeze breakout + 1d ADX trend filter + volume confirmation
# BB Width measures volatility contraction (squeeze). Breakout from squeeze with volume and 1d ADX > 25
# captures explosive moves in both bull and bear markets. Works because low volatility precedes high
# volatility moves, and ADX filters for trending environments to avoid whipsaws in ranges.
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBWidth_Squeeze_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14, np.nan)
    di_minus = np.full_like(tr_14, np.nan)
    valid = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[valid] = (dm_plus_14[valid] / tr_14[valid]) * 100
    di_minus[valid] = (dm_minus_14[valid] / tr_14[valid]) * 100
    
    # DX and ADX
    dx = np.full_like(tr_14, np.nan)
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / 
                    (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    adx_14 = wilders_smoothing(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 6h Bollinger Band Width (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + (2 * std_20)
        lower_bb = ma_20 - (2 * std_20)
        bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    else:
        ma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # BB Width squeeze: width < 20-period percentile 10 (low volatility)
    if len(bb_width) >= 20:
        bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
        bb_width_percentile_10 = pd.Series(bb_width).rolling(window=20, min_periods=20).quantile(0.10).values
        squeeze = bb_width < bb_width_percentile_10
    else:
        squeeze = np.zeros(n, dtype=bool)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long/short on BB Width breakout from squeeze with volume and ADX > 25
            if squeeze[i-1] and not squeeze[i]:  # Breakout from squeeze
                if volume_spike[i] and adx_14_aligned[i] > 25:
                    if close[i] > ma_20[i]:  # Break above middle band = long
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < ma_20[i]:  # Break below middle band = short
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price crosses below middle band OR BB Width expands significantly (end of move)
            if close[i] < ma_20[i] or bb_width[i] > (bb_width_ma_20[i] * 2.0 if not np.isnan(bb_width_ma_20[i]) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle band OR BB Width expands significantly
            if close[i] > ma_20[i] or bb_width[i] > (bb_width_ma_20[i] * 2.0 if not np.isnan(bb_width_ma_20[i]) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals