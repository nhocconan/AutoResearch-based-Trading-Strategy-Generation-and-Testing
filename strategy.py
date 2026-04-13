#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter (ADX + Chop)
    # Long: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND Chop < 61.8 (trending)
    # Short: Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND Chop < 61.8 (trending)
    # Exit: Opposite Elder Ray signal OR regime shifts to range (Chop > 61.8)
    # Uses 6h for lower noise, 1d for regime confirmation, Elder Ray for momentum strength.
    # Discrete sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX(14) and Chop(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range (TR)
    tr_1d = np.zeros(len(close_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # Calculate 1d ADX(14) with min_periods
    adx_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        # +DM and -DM
        plus_dm = np.zeros(len(close_1d))
        minus_dm = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            up_move = high_1d[i] - high_1d[i-1]
            down_move = low_1d[i-1] - low_1d[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        period = 14
        alpha = 1.0 / period
        
        # Initial values (simple average)
        tr_sum = np.sum(tr_1d[:period])
        plus_dm_sum = np.sum(plus_dm[1:period+1])  # Skip first
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        atr = np.full(len(close_1d), np.nan)
        atr[period-1] = tr_sum / period
        plus_di_smoothed = np.full(len(close_1d), np.nan)
        minus_di_smoothed = np.full(len(close_1d), np.nan)
        plus_di_smoothed[period-1] = (plus_dm_sum / period) * 100 / atr[period-1] if atr[period-1] != 0 else 0
        minus_di_smoothed[period-1] = (minus_dm_sum / period) * 100 / atr[period-1] if atr[period-1] != 0 else 0
        
        # Wilder smoothing
        for i in range(period, len(close_1d)):
            atr[i] = (atr[i-1] * (period - 1) + tr_1d[i]) / period
            plus_di_smoothed[i] = (plus_di_smoothed[i-1] * (period - 1) + plus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
            minus_di_smoothed[i] = (minus_di_smoothed[i-1] * (period - 1) + minus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
        
        # DX and ADX
        dx = np.full(len(close_1d), np.nan)
        for i in range(period, len(close_1d)):
            di_sum = plus_di_smoothed[i] + minus_di_smoothed[i]
            dx[i] = abs(plus_di_smoothed[i] - minus_di_smoothed[i]) / di_sum * 100 if di_sum != 0 else 0
        
        # ADX: smoothed DX
        adx_1d[2*period-1] = np.mean(dx[period:2*period])  # Initial ADX
        for i in range(2*period, len(close_1d)):
            adx_1d[i] = (adx_1d[i-1] * (period - 1) + dx[i]) / period
    
    # Calculate 1d Choppiness Index(14)
    chop_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        atr_sum = np.full(len(close_1d), np.nan)
        for i in range(14, len(close_1d)):
            atr_sum[i] = np.sum(tr_1d[i-13:i+1])  # Sum of last 14 TR
        
        max_high = np.full(len(close_1d), np.nan)
        min_low = np.full(len(close_1d), np.nan)
        for i in range(13, len(close_1d)):
            max_high[i] = np.max(high_1d[i-13:i+1])
            min_low[i] = np.min(low_1d[i-13:i+1])
        
        for i in range(14, len(close_1d)):
            if atr_sum[i] > 0 and max_high[i] > min_low[i]:
                chop_1d[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
            else:
                chop_1d[i] = 50.0  # Neutral when undefined
    
    # Align 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    # EMA13 on 6h close
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        ema_13[12] = np.mean(close[:13])
        multiplier = 2 / (13 + 1)
        for i in range(13, n):
            ema_13[i] = (close[i] * multiplier) + (ema_13[i-1] * (1 - multiplier))
    
    bull_power = high - ema_13  # Bull Power
    bear_power = low - ema_13   # Bear Power
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (ADX > 25 and Chop < 61.8)
        trending_regime = (adx_1d_aligned[i] > 25) and (chop_1d_aligned[i] < 61.8)
        
        # Elder Ray signals
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # Entry logic: Trending regime + Elder Ray alignment
        long_entry = trending_regime and bull_power_pos and bear_power_neg
        short_entry = trending_regime and bear_power_neg and bull_power_pos  # Same condition, different interpretation
        
        # Actually, for short we need Bear Power negative AND Bull Power positive (market weak but not strong)
        short_entry = trending_regime and (bear_power[i] < 0) and (bull_power[i] > 0)
        
        # Exit logic: Opposite Elder Ray signal OR regime shifts to range
        long_exit = (bull_power[i] < 0) or (chop_1d_aligned[i] > 61.8)
        short_exit = (bear_power[i] > 0) or (chop_1d_aligned[i] > 61.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_filter_v1"
timeframe = "6h"
leverage = 1.0