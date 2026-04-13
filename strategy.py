#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and 1d ADX trend filter.
    # Long when price breaks above Camarilla H3 + volume > 1.5x 20-period average + ADX > 25 (trending).
    # Short when price breaks below Camarilla L3 + volume > 1.5x 20-period average + ADX > 25 (trending).
    # Exit when price crosses Camarilla pivot point (PP).
    # Uses Camarilla structure from 1d, volume confirmation from 12h, and trend filter from 1d ADX.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3
    # Calculate Camarilla levels
    r4 = pp + ((high_1d - low_1d) * 1.1 / 2)
    r3 = pp + ((high_1d - low_1d) * 1.1 / 4)
    r2 = pp + ((high_1d - low_1d) * 1.1 / 6)
    r1 = pp + ((high_1d - low_1d) * 1.1 / 12)
    s1 = pp - ((high_1d - low_1d) * 1.1 / 12)
    s2 = pp - ((high_1d - low_1d) * 1.1 / 6)
    s3 = pp - ((high_1d - low_1d) * 1.1 / 4)
    s4 = pp - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Get 12h data for volume spike (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d_adx - low_1d_adx)
    tr2 = np.abs(high_1d_adx - np.roll(close_1d_adx, 1))
    tr3 = np.abs(low_1d_adx - np.roll(close_1d_adx, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_1d_adx - np.roll(high_1d_adx, 1)
    down_move = np.roll(low_1d_adx, 1) - low_1d_adx
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align HTF ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirm = vol_12h_aligned[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Entry conditions: breakout + volume + trend
        long_signal = long_breakout and volume_confirm and trend_filter
        short_signal = short_breakout and volume_confirm and trend_filter
        
        # Exit conditions: price crosses pivot point (PP)
        long_exit = close[i] < pp_aligned[i]
        short_exit = close[i] > pp_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_camarilla_vol_adx_breakout_v1"
timeframe = "4h"
leverage = 1.0