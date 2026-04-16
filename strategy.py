#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI (Connors RSI) with 12h ADX trend filter and volume spike confirmation
# Long when CRSI < 15 AND 12h ADX > 25 AND volume > 1.5x 20-period average volume
# Short when CRSI > 85 AND 12h ADX > 25 AND volume > 1.5x 20-period average volume
# Exit when CRSI crosses above 50 (long) or below 50 (short)
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag on 4h timeframe
# CRSI captures short-term mean reversion, ADX ensures trending environment, volume adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === CRSI (Connors RSI) ===
    # RSI(3)
    def rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    rsi_3 = rsi(close, 3)
    
    # RSI of streak
    up_days = np.where(np.diff(close, prepend=close[0]) > 0, 1, 
                      np.where(np.diff(close, prepend=close[0]) < 0, -1, 0))
    streak = np.zeros_like(close)
    for i in range(1, len(close)):
        if up_days[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif up_days[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    abs_streak = np.abs(streak)
    rsi_streak = rsi(abs_streak, 2)
    
    # Percent Rank (100-day)
    def percent_rank(series, window):
        pr = np.zeros_like(series)
        for i in range(len(series)):
            if i < window:
                pr[i] = 50.0
            else:
                window_data = series[i-window:i]
                pr[i] = (np.sum(window_data < series[i]) / window) * 100
        return pr
    percent_rank_100 = percent_rank(close, 100)
    
    # CRSI = (RSI(3) + RSI(streak) + PercentRank(100)) / 3
    crsi = (rsi_3 + rsi_streak + percent_rank_100) / 3.0
    
    # === 12h ADX for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(series, period):
        result = np.zeros_like(series)
        result[period-1] = np.nansum(series[:period])
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[:14])  # First ADX value
    for i in range(14, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 4h Volume Confirmation (20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(crsi[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        crsi_val = crsi[i]
        adx_val = adx_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for confirmation
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when CRSI crosses above 50
            if crsi_val > 50:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when CRSI crosses below 50
            if crsi_val < 50:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: CRSI < 15 AND ADX > 25 AND volume confirmation
            if crsi_val < 15 and adx_val > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: CRSI > 85 AND ADX > 25 AND volume confirmation
            elif crsi_val > 85 and adx_val > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_CRSI_ADX25_Volume1.5x"
timeframe = "4h"
leverage = 1.0