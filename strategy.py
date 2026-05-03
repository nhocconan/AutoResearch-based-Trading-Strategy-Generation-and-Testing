#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
# Long when 6h Williams %R crosses above -80 from below AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period volume MA.
# Short when 6h Williams %R crosses below -20 from above AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period volume MA.
# Exit when Williams %R returns to opposite extreme (-20 for long, -80 for short) or ADX < 20 (range).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Williams %R identifies overbought/oversold conditions, 1d ADX ensures we only trade in trending markets, volume confirms participation.
# Works in both bull and bear markets by only trading reversals from extremes when the higher-timeframe trend is strong.

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[:period] = np.nan
        if len(values) > period:
            result[period] = np.nansum(values[1:period+1])
            for i in range(period+1, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    def williams_r(high_arr, low_arr, close_arr, lookback=14):
        highest_high = np.maximum.accumulate(high_arr)
        lowest_low = np.minimum.accumulate(low_arr)
        # For rolling window, we need to compute properly
        highest_high_rolling = pd.Series(high_arr).rolling(window=lookback, min_periods=lookback).max().values
        lowest_low_rolling = pd.Series(low_arr).rolling(window=lookback, min_periods=lookback).min().values
        wr = np.where((highest_high_rolling - lowest_low_rolling) != 0, 
                      -100 * (highest_high_rolling - close_arr) / (highest_high_rolling - lowest_low_rolling), 
                      -50)  # Neutral when no range
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.5)
        
        # Williams %R conditions
        wr_current = wr[i]
        wr_prev = wr[i-1] if i > 0 else wr[i]
        
        # Williams %R crossing above -80 from below (oversold reversal)
        wr_cross_up = (wr_prev <= -80) and (wr_current > -80)
        # Williams %R crossing below -20 from above (overbought reversal)
        wr_cross_down = (wr_prev >= -20) and (wr_current < -20)
        
        # 1d trend condition: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        # Range condition: ADX < 20 indicates ranging market (exit condition)
        ranging_market = adx_aligned[i] < 20
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND strong trend AND volume spike AND session
            if wr_cross_up and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND strong trend AND volume spike AND session
            elif wr_cross_down and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -20 (overbought) OR market becomes ranging
            if wr_current >= -20 or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -80 (oversold) OR market becomes ranging
            if wr_current <= -80 or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals