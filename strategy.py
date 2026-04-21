#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>1.6x 20-period MA). 
Adds ADX(14) > 20 trend strength filter to reduce whipsaws and overtrading. Target: 60-120 total trades (15-30/year).
Uses discrete sizing (0.25) and ATR(14) stoploss (2.0x). Designed for both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend and Camarilla)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA34 for trend filter (more responsive than EMA50) ===
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d ADX(14) for trend strength filter ===
    # Calculate ADX using 1d data
    high_1d = df_1d_high
    low_1d = df_1d_low
    close_1d = df_1d_close
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smoothing(series, period):
        result = np.zeros_like(series)
        result[period-1] = series[:period].mean()  # first value is simple average
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    tr_14 = wilder_smoothing(tr_1d.values, 14)
    dm_plus_14 = wilder_smoothing(dm_plus.values, 14)
    dm_minus_14 = wilder_smoothing(dm_minus.values, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = np.zeros_like(dx)
    adx_1d[27:] = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values[13:]  # ADX starts after 2*period-1
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (1.6x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        vol_avg = vol_ma[i]
        adx = adx_1d_aligned[i]
        
        # Volume spike: current volume > 1.6x average (balanced for signal frequency)
        volume_spike = volume_now > 1.6 * vol_avg
        # Trend strength: ADX > 20 indicates trending market
        strong_trend = adx > 20
        
        if position == 0:
            # Enter only with volume spike, trend alignment, and strong trend
            long_condition = (price > r1) and (price > ema_34) and volume_spike and strong_trend
            short_condition = (price < s1) and (price < ema_34) and volume_spike and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below EMA)
            elif price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above EMA)
            elif price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0