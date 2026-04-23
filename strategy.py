#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d EMA34 rising AND volume > 1.8x 20-period MA.
Short when price breaks below Donchian lower band AND 1d EMA34 falling AND volume > 1.8x 20-period MA.
Exit when price touches opposite Donchian band or 1d EMA34 reverses.
Uses 1w HTF for regime filter: only trade when 1w ADX < 25 (range market) to avoid whipsaw in strong trends.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian provides clear structure, 1d EMA34 filters major trend, volume confirms breakout strength,
and 1w ADX regime filter reduces trades during strong trends where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w ADX for regime filter (HTF) - only trade in ranging markets (ADX < 25)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20, 14+14+14)  # Donchian, EMA34, volume MA, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        adx_val = adx_1w_aligned[i]
        
        # Regime filter: only trade when 1w ADX < 25 (ranging market)
        regime_filter = adx_val < 25
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA34 rising AND volume filter AND regime filter
            if price > upper and ema_rising and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA34 falling AND volume filter AND regime filter
            elif price < lower and ema_falling and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite) OR EMA34 starts falling OR regime changes (ADX >= 25)
                if price < lower or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]) or adx_val >= 25:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite) OR EMA34 starts rising OR regime changes (ADX >= 25)
                if price > upper or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]) or adx_val >= 25:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ADXRegime"
timeframe = "12h"
leverage = 1.0