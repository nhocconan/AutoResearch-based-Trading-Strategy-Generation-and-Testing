#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter_v1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 1h with 4h EMA50 trend filter and 1d volume spike confirmation.
Uses 4h EMA50 for intermediate trend to reduce whipsaws vs faster MA, and 1d 2.0x volume spike for institutional confirmation.
Only trade in trending markets (ADX > 20 on 4h) to avoid chop. Designed for 15-35 trades/year on 1h timeframe.
Works in bull/bear markets by following 4h EMA50 trend and filtering ranging regimes via ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter (intermediate trend for fewer whipsaws)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 4h OHLC
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_r1 = prev_close_4h + 1.125 * (prev_high_4h - prev_low_4h)
    camarilla_s1 = prev_close_4h - 1.125 * (prev_high_4h - prev_low_4h)
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume median (20-period) for institutional volume confirmation
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # ADX(14) for regime filter on 4h - trending when > 20
    # Calculate ADX using 4h data
    plus_dm = np.zeros(len(df_4h))
    minus_dm = np.zeros(len(df_4h))
    tr = np.zeros(len(df_4h))
    
    for i in range(1, len(df_4h)):
        plus_dm[i] = max(df_4h['high'].iloc[i] - df_4h['high'].iloc[i-1], 0) if (df_4h['high'].iloc[i] - df_4h['high'].iloc[i-1]) > (df_4h['low'].iloc[i-1] - df_4h['low'].iloc[i]) else 0
        minus_dm[i] = max(df_4h['low'].iloc[i-1] - df_4h['low'].iloc[i], 0) if (df_4h['low'].iloc[i-1] - df_4h['low'].iloc[i]) > (df_4h['high'].iloc[i] - df_4h['high'].iloc[i-1]) else 0
        tr[i] = max(df_4h['high'].iloc[i] - df_4h['low'].iloc[i], abs(df_4h['high'].iloc[i] - df_4h['close'].iloc[i-1]), abs(df_4h['low'].iloc[i] - df_4h['close'].iloc[i-1]))
    
    # Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(df_4h) >= period:
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        tr_smooth = WilderSmooth(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1)
        adx_4h = WilderSmooth(dx, period)
    else:
        adx_4h = np.full(len(df_4h), np.nan)
    
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Session filter: 08-20 UTC (reduces noise trades outside active sessions)
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 4h EMA (50), 1d volume median (20), ADX (14*2 for smoothing)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_4h_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten or hold flat
            signals[i] = 0.0
            position = 0
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        adx_val = adx_4h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend, and trending regime
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_1d_val) and \
                          (close_val > ema_50_4h_val) and \
                          (adx_val > 20)
            
            # Short: break below S1 with volume spike, downtrend, and trending regime
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_1d_val) and \
                           (close_val < ema_50_4h_val) and \
                           (adx_val > 20)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 (reversal) or trend changes (close < 4h EMA50) or regime changes (ADX < 15)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_4h_val) or \
               (adx_val < 15):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 (reversal) or trend changes (close > 4h EMA50) or regime changes (ADX < 15)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_4h_val) or \
               (adx_val < 15):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter_v1"
timeframe = "1h"
leverage = 1.0