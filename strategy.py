#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Chop_Filter_v1
Hypothesis: On 12h timeframe, KAMA direction combined with RSI momentum and Choppiness regime filter provides robust trend-following signals that work in both bull and bear markets. The Choppiness filter avoids whipsaws in ranging markets, while KAMA adapts to changing volatility. This strategy targets 20-50 trades per year to minimize fee drag and improve generalization across BTC, ETH, and SOL.
"""
name = "12h_KAMA_Trend_RSI_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle the array operations properly
    change_padded = np.concatenate([np.full(er_period-1, np.nan), change])
    volatility_padded = np.concatenate([np.full(er_period-1, np.nan), volatility[er_period-1:]])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smooth ER
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period-1] = close[er_period-1]
    for i in range(er_period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Close for trend filter
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index (14) - needs high/low
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    
    # ATR(14)
    atr = np.full_like(close, np.nan)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    sum_tr = np.full_like(close, np.nan)
    for i in range(14, n):
        if i == 14:
            sum_tr[i] = np.sum(tr[1:15])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-14] + tr[i]
    
    # Choppiness Index
    # Highest high and lowest low over 14 periods
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.full_like(close, np.nan)
    for i in range(14, n):
        if sum_tr[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Align 1d trend filter
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = np.full_like(volume, np.nan)
    for i in range(20, n):
        if i == 20:
            vol_avg[i] = np.mean(volume[0:20])
        else:
            vol_avg[i] = vol_avg[i-1] - volume[i-20]/20 + volume[i]/20
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price > KAMA + RSI > 50 + Chop < 61.8 (trending) + volume
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price < KAMA + RSI < 50 + Chop < 61.8 (trending) + volume
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Minimum holding period of 2 bars to reduce trade frequency
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit: reverse signal or chop becomes too high (ranging market)
            if position == 1:
                if close[i] < kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals