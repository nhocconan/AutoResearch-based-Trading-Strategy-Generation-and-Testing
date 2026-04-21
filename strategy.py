#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d ADX Trend Filter + Volume Spike
# Long when Williams %R crosses above -20 from below and 1d ADX > 25 and 1d volume > 1.5x 20-period average
# Short when Williams %R crosses below -80 from above and 1d ADX > 25 and 1d volume > 1.5x 20-period average
# Exit when Williams %R crosses back below -80 (for long) or above -20 (for short)
# Williams %R identifies overbought/oversold conditions, ADX filters for trending markets,
# Volume confirms momentum. Target: 20-35 trades/year by requiring strict alignment of conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    
    williams_r = -100 * (highest_high - close_1d) / rr
    
    # Calculate 1d ADX(14) for trend strength
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial smoothed values
    tr_sum[tr_period-1] = np.nansum(tr[:tr_period])
    dm_plus_sum[tr_period-1] = np.nansum(dm_plus[:tr_period])
    dm_minus_sum[tr_period-1] = np.nansum(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx) | np.isinf(dx)] = 0
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.nanmean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams %R alignment
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current 1d values (already aligned)
        williams_r_val = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Get current 1d volume for spike detection
        vol_idx = min(i // 96, len(df_1d) - 1)  # 96 = 24*60/15 (4h bars per day)
        current_vol = df_1d['volume'].iloc[vol_idx] if vol_idx < len(df_1d) else df_1d['volume'].iloc[-1]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirm = current_vol > 1.5 * vol_ma_val
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            if volume_confirm and trend_filter:
                # Long: Williams %R crosses above -20 from below (exiting oversold)
                if i > 14 and williams_r_val > -20 and williams_r_aligned[i-1] <= -20:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 from above (exiting overbought)
                elif i > 14 and williams_r_val < -80 and williams_r_aligned[i-1] >= -80:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses back below -80 (re-entering overbought)
                if i > 14 and williams_r_val < -80 and williams_r_aligned[i-1] >= -80:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses back above -20 (re-entering oversold)
                if i > 14 and williams_r_val > -20 and williams_r_aligned[i-1] <= -20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dADX_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0