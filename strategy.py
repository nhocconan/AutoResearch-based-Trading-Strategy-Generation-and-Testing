#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume spike confirmation
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper BB AND ADX(1d) > 25 AND volume > 2.0x 20-period average
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower BB AND ADX(1d) > 25 AND volume > 2.0x 20-period average
# Exit when price returns to middle BB (20-period SMA) OR ADX(1d) < 20 (trend weakens)
# Bollinger Bands capture volatility contraction/expansion cycles
# ADX(1d) ensures we only trade in trending regimes on higher timeframe
# Volume spike confirms institutional participation in breakout
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_BB_Squeeze_Breakout_1dADX25_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    tr_smooth = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands on 6h data (20, 2)
    bb_period = 20
    bb_std = 2
    if len(close) >= bb_period:
        bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
        bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
        bb_upper = bb_ma + (bb_std_dev * bb_std)
        bb_lower = bb_ma - (bb_std_dev * bb_std)
        bb_width = bb_upper - bb_lower
        
        # BB width percentile for squeeze detection (20th percentile lookback)
        bb_width_percentile = np.zeros_like(bb_width)
        for i in range(bb_period, len(bb_width)):
            if i >= 50:  # Need sufficient lookback for percentile
                lookback = bb_width[max(0, i-50):i+1]
                valid_vals = lookback[~np.isnan(lookback)]
                if len(valid_vals) >= 10:
                    percentile_20 = np.percentile(valid_vals, 20)
                    bb_width_percentile[i] = (bb_width[i] <= percentile_20) * 1.0
                else:
                    bb_width_percentile[i] = 0
            else:
                bb_width_percentile[i] = 0
    else:
        bb_ma = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        bb_width_percentile = np.zeros(n)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bb_ma[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: BB squeeze AND price breaks above upper BB AND ADX > 25 AND volume spike
            if (bb_width_percentile[i] > 0.5 and  # True when in squeeze (width <= 20th percentile)
                close[i] > bb_upper[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: BB squeeze AND price breaks below lower BB AND ADX > 25 AND volume spike
            elif (bb_width_percentile[i] > 0.5 and  # True when in squeeze
                  close[i] < bb_lower[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB OR ADX < 20 (trend weakens)
            if (close[i] < bb_ma[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB OR ADX < 20 (trend weakens)
            if (close[i] > bb_ma[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals