#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend following with daily volume confirmation and weekly volatility regime filter
# KAMA adapts to market conditions - fast in trends, slow in ranges
# Long when KAMA slope > 0 with volume > 1.5x average and weekly volatility < 40th percentile
# Short when KAMA slope < 0 with volume > 1.5x average and weekly volatility < 40th percentile
# Exit when KAMA slope reverses or volatility exceeds 60th percentile
# Targets 25-35 trades/year to minimize fee decay while capturing sustained trends

name = "12h_KAMA_Vol_VolRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Get weekly data for volatility regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    # Recalculate volatility properly
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # incorrect, redo
    
    # Proper ER calculation
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # still wrong
    
    # Correct approach: calculate ER for each point
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        change = np.abs(close[i] - close[i-10])
        volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_30 = np.full_like(daily_volume, np.nan)
    for i in range(len(daily_volume)):
        if i < 30:
            vol_ma_30[i] = np.mean(daily_volume[max(0, i-29):i+1]) if i >= 0 else daily_volume[i]
        else:
            vol_ma_30[i] = np.mean(daily_volume[i-29:i+1])
    
    # Calculate weekly volatility percentile (using ATR-based volatility)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR10
    atr10 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 10:
            atr10[i] = np.mean(tr[max(0, i-9):i+1]) if i >= 0 else tr[i]
        else:
            atr10[i] = np.mean(tr[i-9:i+1])
    
    # Volatility percentile rank (using 50-period lookback)
    vol_rank = np.full_like(atr10, np.nan)
    for i in range(50, len(atr10)):
        window = atr10[i-50:i+1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            current = atr10[i]
            if not np.isnan(current):
                # Calculate percentile rank
                rank = np.sum(~np.isnan(window) & (window <= current)) / np.sum(~np.isnan(window)) * 100
                vol_rank[i] = rank
    
    # Align indicators to 12h timeframe
    kama_aligned = kama  # already on 12h timeframe
    vol_ma_30_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_30)
    vol_rank_aligned = align_htf_to_ltf(prices, df_weekly, vol_rank)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # warmup for KAMA and vol rank
    
    for i in range(start_idx, n):
        if (np.isnan(vol_ma_30_aligned[i]) or np.isnan(vol_rank_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for volume filter
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 30-day average
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.5 * vol_ma_30_aligned[i]
        
        # Volatility regime: < 40th percentile = low volatility (trending)
        vol_regime = vol_rank_aligned[i] < 40
        
        if position == 0:
            # Look for KAMA direction with volume confirmation and low volatility regime
            # Long: KAMA sloping up
            if i > 0 and not np.isnan(kama_aligned[i]) and not np.isnan(kama_aligned[i-1]):
                if kama_aligned[i] > kama_aligned[i-1]:
                    if vol_filter and vol_regime:
                        signals[i] = 0.25
                        position = 1
                # Short: KAMA sloping down
                elif kama_aligned[i] < kama_aligned[i-1]:
                    if vol_filter and vol_regime:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: KAMA slopes down or volatility exceeds 60th percentile
            if i > 0 and not np.isnan(kama_aligned[i]) and not np.isnan(kama_aligned[i-1]):
                if kama_aligned[i] < kama_aligned[i-1] or vol_rank_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA slopes up or volatility exceeds 60th percentile
            if i > 0 and not np.isnan(kama_aligned[i]) and not np.isnan(kama_aligned[i-1]):
                if kama_aligned[i] > kama_aligned[i-1] or vol_rank_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals