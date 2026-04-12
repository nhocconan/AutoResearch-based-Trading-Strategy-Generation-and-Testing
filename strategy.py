#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: On 12h timeframe, use KAMA (Kaufman Adaptive Moving Average) to capture the dominant trend direction,
filtered by RSI(14) to avoid overextended entries, with volume confirmation and ADX(14) trend strength filter.
Exit when price crosses KAMA in the opposite direction. Designed for low trade frequency (10-30/year) by requiring
multiple confluence factors. Works in bull/bear via adaptive trend following and RSI filter to avoid chasing momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_Trend_With_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY KAMA (10-period ER, 2/30 smoothing) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)))
    
    # Initialize arrays
    er = np.full_like(close_1d, np.nan, dtype=np.float64)
    sc = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    
    # Calculate ER and SC for 10-period window
    for i in range(10, len(close_1d)):
        if i >= 10:
            change_val = np.abs(close_1d[i] - close_1d[i-10])
            volatility_val = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 1.0
            sc[i] = (er[i] * (2/2 - 2/30) + 2/30) ** 2  # Fast=2, Slow=30
    
    # Initialize KAMA
    kama[9] = close_1d[9]  # Start at period 9
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === DAILY RSI(14) ===
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    rsi = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing
    for i in range(1, len(close_1d)):
        if i < 14:
            if i == 1:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # === DAILY ADX(14) ===
    # Calculate True Range and Directional Movement
    tr = np.full_like(close_1d, np.nan)
    plus_dm = np.full_like(close_1d, np.nan)
    minus_dm = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    atr = np.full_like(close_1d, np.nan)
    smoothed_plus_dm = np.full_like(close_1d, np.nan)
    smoothed_minus_dm = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if i < 14:
            if i == 1:
                atr[i] = tr[i]
                smoothed_plus_dm[i] = plus_dm[i]
                smoothed_minus_dm[i] = minus_dm[i]
            else:
                atr[i] = (atr[i-1] * (i-1) + tr[i]) / i
                smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (i-1) + plus_dm[i]) / i
                smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (i-1) + minus_dm[i]) / i
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * 13 + plus_dm[i]) / 14
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI and DX
    plus_di = np.full_like(close_1d, np.nan)
    minus_di = np.full_like(close_1d, np.nan)
    dx = np.full_like(close_1d, np.nan)
    adx = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if atr[i] != 0:
            plus_di[i] = 100 * (smoothed_plus_dm[i] / atr[i])
            minus_di[i] = 100 * (smoothed_minus_dm[i] / atr[i])
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    for i in range(28, len(close_1d)):  # 14 + 14
        if i == 28:
            adx[i] = np.nanmean(dx[14:i+1])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # === DAILY VOLUME AVERAGE (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full_like(volume_1d, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(volume_1d)):
        vol_sum += volume_1d[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume_1d[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg_1d[i] = vol_sum / vol_count
    
    # Align all daily indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i]) or vol_avg_aligned[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Trend strength filter: ADX > 20
        trend_filter = adx_aligned[i] > 20
        
        # RSI filter: avoid overextended (30 < RSI < 70)
        rsi_filter = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Entry conditions
        long_setup = (close[i] > kama_aligned[i]) and vol_confirm and trend_filter and rsi_filter
        short_setup = (close[i] < kama_aligned[i]) and vol_confirm and trend_filter and rsi_filter
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = close[i] < kama_aligned[i]
        exit_short = close[i] > kama_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals