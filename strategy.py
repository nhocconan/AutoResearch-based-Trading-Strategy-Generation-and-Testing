#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout + 1d Volume Spike + 1d ADX Trend Filter
# Long when price breaks above Camarilla R3 and 1d volume > 1.5x 20-period average and 1d ADX > 20
# Short when price breaks below Camarilla S3 and 1d volume > 1.5x 20-period average and 1d ADX > 20
# Exit when price crosses Camarilla pivot (central level)
# Uses Camarilla levels from daily pivot for institutional reference points
# Volume confirms breakout strength, ADX filters for trending markets
# Target: 20-35 trades/year by requiring multiple confirmations

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
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    #          S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Pivot (close) = (H+L+C)/3
    typical_price = (high_1d + low_1d + close_1d) / 3
    camarilla_pivot = typical_price
    range_hl = high_1d - low_1d
    
    camarilla_r3 = camarilla_pivot + (range_hl * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_hl * 1.1 / 4)
    camarilla_r4 = camarilla_pivot + (range_hl * 1.1 / 2)
    camarilla_s4 = camarilla_pivot - (range_hl * 1.1 / 2)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = df_1d['volume'].iloc[i // 96] > 1.5 * vol_ma if i >= 96 else df_1d['volume'].iloc[0] > 1.5 * vol_ma
        
        # Trend filter: ADX > 20 indicates trending market
        trend_filter = adx_aligned[i] > 20
        
        if position == 0:
            if volume_confirm and trend_filter:
                # Long: price breaks above Camarilla R3
                if price > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Camarilla S3
                elif price < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Camarilla pivot
            exit_signal = False
            
            if position == 1:  # long position
                if price < camarilla_pivot_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > camarilla_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dADX_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0