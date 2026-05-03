#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme levels with 1d ADX trend filter and volume spike confirmation.
# Long when 4h Williams %R crosses above -20 from below (extreme short-term oversold) AND 1d ADX > 25 (strong trend) AND 4h volume > 1.5x 20-period volume MA.
# Short when 4h Williams %R crosses below -80 from above (extreme short-term overbought) AND 1d ADX > 25 (strong trend) AND 4h volume > 1.5x 20-period volume MA.
# Exit when Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakness).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Williams %R captures short-term exhaustion, ADX confirms trend strength for follow-through, volume validates participation.
# Works in both bull and bear markets by trading mean reversions within strong trends when volume confirms.

name = "4h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_Session"
timeframe = "4h"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Williams %R
    highest_high_4h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_4h = np.where((highest_high_4h - lowest_low_4h) != 0, 
                             -100 * (highest_high_4h - close) / (highest_high_4h - lowest_low_4h), -50)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r_4h[i]) or 
            np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Williams %R conditions
        wr_current = williams_r_4h[i]
        wr_prev = williams_r_4h[i-1]
        
        # Cross above -20 from below (oversold bounce)
        wr_cross_up = (wr_prev <= -20) and (wr_current > -20)
        # Cross below -80 from above (overbought rejection)
        wr_cross_down = (wr_prev >= -80) and (wr_current < -80)
        
        # Volume spike condition: current 4h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 1.5)
        
        # 1d ADX trend condition: ADX > 25 for strong trend
        strong_trend = adx_1d_aligned[i] > 25
        # Weak trend condition for exit: ADX < 20
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: Williams %R cross above -20 AND strong trend AND volume spike AND session
            if wr_cross_up and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R cross below -80 AND strong trend AND volume spike AND session
            elif wr_cross_down and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR trend weakens
            if wr_current >= -50 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR trend weakens
            if wr_current <= -50 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals