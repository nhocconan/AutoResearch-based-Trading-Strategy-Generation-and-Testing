#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ADX (14-period) for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr[i] = tr[i]
                dm_plus_smooth[i] = dm_plus[i]
                dm_minus_smooth[i] = dm_minus[i]
            else:
                atr[i] = (atr[i-1] * (i-1) + tr[i]) / i
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (i-1) + dm_plus[i]) / i
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (i-1) + dm_minus[i]) / i
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < period:
            if i == 0:
                adx[i] = dx[i]
            else:
                adx[i] = (adx[i-1] * (i-1) + dx[i]) / i
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # === 1d 200-period EMA for trend direction ===
    ema_200 = np.full_like(close_1d, np.nan)
    alpha = 2 / (200 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_200[i] = close_1d[i]
        elif np.isnan(ema_200[i-1]):
            ema_200[i] = close_1d[i]
        else:
            ema_200[i] = alpha * close_1d[i] + (1 - alpha) * ema_200[i-1]
    
    # === Align 1d indicators to 4h timeframe ===
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 4h Donchian Channel (20-period) ===
    # Highest high of last 20 periods
    highest_high = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[max(0, i-9):i+1])
        else:
            highest_high[i] = high[i]
    
    # Lowest low of last 20 periods
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(low)):
        if i >= 19:
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            lowest_low[i] = np.min(low[max(0, i-9):i+1])
        else:
            lowest_low[i] = low[i]
    
    # === 4h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: ADX > 25 and price above/below EMA200
        is_strong_uptrend = adx_aligned[i] > 25 and close[i] > ema_200_aligned[i]
        is_strong_downtrend = adx_aligned[i] > 25 and close[i] < ema_200_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: strong uptrend + price breaks above Donchian high + volume confirmation
            if is_strong_uptrend and close[i] > highest_high[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: strong downtrend + price breaks below Donchian low + volume confirmation
            elif is_strong_downtrend and close[i] < lowest_low[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: trend weakens OR price retests Donchian low
            if adx_aligned[i] < 20 or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens OR price retests Donchian high
            if adx_aligned[i] < 20 or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Donchian_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0