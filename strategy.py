#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 Breakout + 1d Volume Spike + 1d ADX Trend Filter
# Long when price breaks above Camarilla R1 and 1d volume > 2x 20-period average and 1d ADX > 25
# Short when price breaks below Camarilla S1 and 1d volume > 2x 20-period average and 1d ADX > 25
# Exit when price crosses Camarilla pivot point (PP)
# Uses Camarilla levels from daily timeframe for structure, volume spike for confirmation, ADX for trend filter
# Target: 20-35 trades/year by requiring ADX > 25 + volume spike + Camarilla breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
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
    
    # Calculate Camarilla pivot levels from 1d data
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (high_1d + low_1d + close_1d) / 3
    pp = typical_price
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        # Get current 1d volume (approximate for 4h bar)
        idx_1d = i // 6  # 6 four-hour bars in a day
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        vol_current = df_1d['volume'].iloc[idx_1d]
        volume_confirm = vol_current > 2 * vol_ma
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            if volume_confirm and trend_filter:
                # Long: price breaks above Camarilla R1
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Camarilla S1
                elif price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Camarilla pivot point (PP)
            exit_signal = False
            
            if position == 1:  # long position
                if price < pp_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dADX_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0