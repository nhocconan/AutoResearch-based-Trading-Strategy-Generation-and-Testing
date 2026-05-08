#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Weekly Pivot + ADX Trend + Volume Breakout
# Uses weekly pivot points (more robust than daily) for key support/resistance
# ADX(14) > 25 ensures we only trade in trending markets
# Volume > 1.5x average confirms breakout validity
# Works in bull via breakouts above weekly R1, in bear via breakdowns below weekly S1
# Target: 15-25 trades/year to minimize fee drag while capturing major moves
name = "6h_WeeklyPivot_ADXTrend_VolumeBreakout"
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
    
    # Get weekly data for pivot points (more robust than daily)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    
    # Align weekly data to 6h timeframe
    prev_week_high_aligned = align_htf_to_ltf(prices, df_w, prev_week_high)
    prev_week_low_aligned = align_htf_to_ltf(prices, df_w, prev_week_low)
    prev_week_close_aligned = align_htf_to_ltf(prices, df_w, prev_week_close)
    
    # Calculate pivot points
    pivot = (prev_week_high_aligned + prev_week_low_aligned + prev_week_close_aligned) / 3.0
    range_ = prev_week_high_aligned - prev_week_low_aligned
    
    # Weekly R1 and S1 (primary levels)
    r1 = 2 * pivot - prev_week_low_aligned
    s1 = 2 * pivot - prev_week_high_aligned
    
    # Get daily data for ADX trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above weekly R1 with strong trend and volume
            if (close[i] > r1[i] and 
                adx_aligned[i] > 25 and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly S1 with strong trend and volume
            elif (close[i] < s1[i] and 
                  adx_aligned[i] > 25 and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below weekly S1 (mean reversion) OR trend weakens
            if close[i] < s1[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above weekly R1 (mean reversion) OR trend weakens
            if close[i] > r1[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals