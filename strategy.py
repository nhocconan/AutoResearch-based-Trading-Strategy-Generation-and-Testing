#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20 to -80 range)
# Reversal signals: %R crosses above -80 from below (long) or below -20 from above (short)
# ADX > 25 confirms trend strength - only trade in direction of 1d trend
# Volume spike (>1.5x average) confirms momentum behind the reversal
# Works in bull/bear markets by fading extremes in ranging markets and catching pullbacks in trends
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_WilliamsR_Reversal_1dADXTrend_Volume"
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
    
    # Williams %R (14 period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14 period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period]) if not np.any(np.isnan(values[:period])) else np.nan
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    # Handle division by zero
    di_plus = np.where(tr14 == 0, 0, di_plus)
    di_minus = np.where(tr14 == 0, 0, di_minus)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d EMA34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for Williams %R and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below (oversold reversal)
            # with bullish trend (price > EMA34) and volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                close[i] > ema_34_1d_aligned[i] and
                adx_14_1d_aligned[i] > 25 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought reversal)
            # with bearish trend (price < EMA34) and volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  close[i] < ema_34_1d_aligned[i] and
                  adx_14_1d_aligned[i] > 25 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend turns bearish
            if williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend turns bullish
            if williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals