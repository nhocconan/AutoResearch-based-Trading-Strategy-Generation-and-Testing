#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d ADX trend filter and volume spike confirmation
# Uses 1d Camarilla pivot levels (H3/L3) as entry triggers in direction of 1d ADX > 25
# Volume confirmation requires current volume > 2.0x 24-period average to avoid false breakouts
# Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: ADX filter ensures we only trend-follow when trend is strong, avoiding whipsaws in ranging markets

name = "12h_1d_camarilla_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H3 = PP + (H - L) * 1.1 / 4
    # L3 = PP - (H - L) * 1.1 / 4
    h3 = pp + (range_1d * 1.1 / 4.0)
    l3 = pp - (range_1d * 1.1 / 4.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d ADX (14-period) for trend strength filter
    # +DM = max(H[i] - H[i-1], 0) if H[i] - H[i-1] > L[i-1] - L[i] else 0
    # -DM = max(L[i-1] - L[i], 0) if L[i-1] - L[i] > H[i] - H[i-1] else 0
    # TR = max(H[i] - L[i], H[i-1] - C[i-1], L[i-1] - C[i-1])
    # smoothed +DM, -DM, TR using Wilder's smoothing (alpha = 1/period)
    # DI+ = 100 * smoothed +DM / smoothed TR
    # DI- = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    # ADX = smoothed DX
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is undefined
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing (EMA with alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for volume spike confirmation (12h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 24:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2.0x 24-period average
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR ADX < 20 (trend weakening)
            if close[i] < l3_12h[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR ADX < 20 (trend weakening)
            if close[i] > h3_12h[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume spike and ADX > 25 (strong trend)
            if volume_spike and adx_12h[i] > 25:
                # Long entry: price closes above H3 (bullish breakout)
                if close[i] > h3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes below L3 (bearish breakout)
                elif close[i] < l3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals