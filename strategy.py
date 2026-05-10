#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v4
# Hypothesis: Breakout above 12h Camarilla R1 or below S1 with volume surge and 1d EMA34 trend confirmation.
# Optimized for lower trade frequency: increased volume threshold to 2.0x and added ADX filter (ADX>25) to confirm trend strength.
# Targets 10-20 trades/year. Works in bull/bear by requiring strong trend alignment, reducing false breakouts.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v4"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLC for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels using previous period's range
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    camarilla_s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Volume average (20-period = ~10 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Camarilla (needs 1 bar) + EMA34 (34) + ADX (14+14=28) + volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (2.0x average - increased threshold for fewer trades)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Trend strength filter (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Breakout above Camarilla R1 or breakdown below S1
        breakout_r1 = close[i] > camarilla_r1[i]
        breakdown_s1 = close[i] < camarilla_s1[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above Camarilla R1 with volume surge, strong trend, and 1d uptrend
            if breakout_r1 and volume_surge and strong_trend and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Breakdown below Camarilla S1 with volume surge, strong trend, and 1d downtrend
            elif breakdown_s1 and volume_surge and strong_trend and downtrend:
                signals[i] = -0.30
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (36 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Camarilla S1 or trend changes
                if close[i] < camarilla_s1[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Short exit: price breaks above Camarilla R1 or trend changes
                if close[i] > camarilla_r1[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals