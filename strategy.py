#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R4/S4 breakout with 1d ADX25 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R4 AND 1d ADX > 25 (strong trend) AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below 12h Camarilla S4 AND 1d ADX > 25 (strong trend) AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back through the 12h Camarilla midpoint (R4/S4 average)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Camarilla R4/S4 levels provide wider breakout structure that reduces whipsaw in choppy markets
# 1d ADX > 25 filter ensures we only trade during strong trending regimes, reducing false breakouts
# Volume confirmation (2.0x) validates breakout strength with higher threshold to avoid overtrading

name = "4h_12hCamarillaR4S4_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least one completed 12h bar
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (R4, S4, midpoint)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    high_low_12h = high_12h - low_12h
    camarilla_r4_12h = close_12h + 1.1 * high_low_12h * 1.1 / 2.0
    camarilla_s4_12h = close_12h - 1.1 * high_low_12h * 1.1 / 2.0
    camarilla_mid_12h = (camarilla_r4_12h + camarilla_s4_12h) / 2.0
    
    # Align 12h Camarilla to 4h timeframe (wait for completed 12h bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid_12h)
    
    # Get 1d data ONCE before loop for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip first NaN in tr
        # Wilder's smoothing: previous * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
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
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R4, 1d ADX > 25 (strong trend), volume confirmation, in session
            if (close[i] > camarilla_r4_aligned[i] and 
                adx_1d_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 12h Camarilla S4, 1d ADX > 25 (strong trend), volume confirmation, in session
            elif (close[i] < camarilla_s4_aligned[i] and 
                  adx_1d_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above 12h Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals