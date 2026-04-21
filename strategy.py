#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot + 1d Volume Spike + 1d ADX Trend Filter
# Long when price touches S1 or S2 with bullish reversal candle and 1d volume > 1.5x average and ADX > 20
# Short when price touches R1 or R2 with bearish reversal candle and 1d volume > 1.5x average and ADX > 20
# Exit when price crosses the pivot point (PP)
# Camarilla levels provide precise support/resistance in ranging markets
# Volume confirms institutional interest
# ADX ensures we avoid whipsaws in weak trends
# Target: 25-40 trades/year by requiring confluence of level, volume, and trend

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
    
    # Calculate Camarilla levels from previous day
    # Typical Price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    
    # Shift by 1 to use previous day's data for today's levels
    pp = np.roll(typical_price, 1)  # Pivot Point
    r1 = pp + (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    s1 = pp - (range_1d * 1.1 / 12)
    s2 = pp - (range_1d * 1.1 / 6)
    
    # Handle first value
    pp[0] = typical_price[0]
    r1[0] = pp[0] + (range_1d[0] * 1.1 / 12)
    r2[0] = pp[0] + (range_1d[0] * 1.1 / 6)
    s1[0] = pp[0] - (range_1d[0] * 1.1 / 12)
    s2[0] = pp[0] - (range_1d[0] * 1.1 / 6)
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        vol_ratio = df_1d['volume'].iloc[i // 96] / vol_ma if i >= 96 and vol_ma > 0 else 1.0
        volume_confirm = vol_ratio > 1.5
        
        # Trend filter: ADX > 20 indicates sufficient trend strength
        trend_filter = adx_aligned[i] > 20
        
        # Reversal candle detection
        is_bullish = prices['close'].iloc[i] > prices['open'].iloc[i]
        is_bearish = prices['close'].iloc[i] < prices['open'].iloc[i]
        
        if position == 0:
            if volume_confirm and trend_filter:
                # Long: price touches S1 or S2 with bullish candle
                if (abs(price - s1_aligned[i]) < 0.001 * price or abs(price - s2_aligned[i]) < 0.001 * price) and is_bullish:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches R1 or R2 with bearish candle
                elif (abs(price - r1_aligned[i]) < 0.001 * price or abs(price - r2_aligned[i]) < 0.001 * price) and is_bearish:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses pivot point
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

name = "4h_Camarilla_Pivot_1dVolume_ADX_Trend"
timeframe = "4h"
leverage = 1.0