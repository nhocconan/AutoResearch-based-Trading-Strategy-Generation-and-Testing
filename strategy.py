#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and ADX regime filter.
Long when price breaks above R1 AND volume > 2.0x 20-period average AND ADX > 25 (trending).
Short when price breaks below S1 AND volume > 2.0x 20-period average AND ADX > 25 (trending).
Exit when price crosses the 50-period EMA on 4h.
Camarilla levels provide intraday support/resistance, volume confirms breakout validity,
ADX ensures we only trade in trending markets to avoid chop, EMA exit provides clean trend-following exit.
Designed to work in both bull and bear markets by trading with the trend via ADX filter.
Targets 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 4h data for price structure
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng / 12
    camarilla_s1 = close_1d - 1.1 * rng / 12
    
    # Get 1d data for volume spike filter
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 4h for regime filter (trending vs ranging)
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # +DI = 100 * smoothed(+DM) / smoothed(TR)
    # -DI = 100 * smoothed(-DM) / smoothed(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed(DX)
    
    # Calculate True Range (TR)
    prev_close = np.roll(close_4h, 1)
    prev_close[0] = close_4h[0]
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - prev_close)
    tr3 = np.abs(low_4h - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    prev_high = np.roll(high_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low = np.roll(low_4h, 1)
    prev_low[0] = low_4h[0]
    
    up_move = high_4h - prev_high
    down_move = prev_low - low_4h
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothing with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # Calculate DX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Calculate ADX (smoothed DX)
    adx = wilder_smooth(dx, period)
    
    # Calculate 50-period EMA on 4h for exit
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators (ADX + EMA50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        ema_50 = ema_50_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 2.0x avg AND ADX > 25 (trending)
            if high_price > r1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 2.0x avg AND ADX > 25 (trending)
            elif low_price < s1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 50 EMA
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 50 EMA
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dVolume_ADXFilter"
timeframe = "4h"
leverage = 1.0