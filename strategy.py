#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour EMA TREND FOLLOWING + DAILY VOLUME CONFIRMATION + WEEKLY CHOPPINESS REGIME FILTER
# Long when price > EMA50 with volume confirmation and weekly chop < 40 (trending)
# Short when price < EMA50 with volume confirmation and weekly chop < 40 (trending)
# Exit when price crosses EMA50 or weekly chop > 50 (range)
# EMA50 provides smooth trend following, volume confirms momentum, chop filter avoids whipsaws
# Targets 20-30 trades/year to minimize fee decay while capturing sustained trends

name = "12h_EMA50_Volume_Chop"
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
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Get weekly data for chop regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly Choppiness Index (14)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR14
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high14 = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14)/(max_high-min_low)) / log10(14)
    # Avoid division by zero
    range_14 = max_high14 - min_low14
    chop = np.zeros_like(range_14, dtype=float)
    mask = (range_14 > 0) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(sum_atr14[mask] / range_14[mask]) / np.log10(14)
    # For invalid cases, set to 50 (neutral)
    chop[~mask] = 50
    
    # Align indicators to 12h timeframe
    ema50_aligned = ema50  # already on 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_weekly, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
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
        
        # Volume filter: current daily volume > 1.3x 20-day EMA
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for EMA crossover with volume confirmation and trending regime
            # Long: price crosses above EMA50
            if close[i] > ema50_aligned[i] and close[i-1] <= ema50_aligned[i-1]:
                if vol_filter and chop_aligned[i] < 40:
                    signals[i] = 0.25
                    position = 1
            # Short: price crosses below EMA50
            elif close[i] < ema50_aligned[i] and close[i-1] >= ema50_aligned[i-1]:
                if vol_filter and chop_aligned[i] < 40:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below EMA50 or chop > 50 (range)
            if close[i] < ema50_aligned[i] or chop_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA50 or chop > 50 (range)
            if close[i] > ema50_aligned[i] or chop_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals