#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points from previous day (complete day only)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's complete data to calculate today's pivot
    # Skip the current incomplete day
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    prev_close = close_1d[:-1]
    
    # Need at least one complete day
    if len(prev_high) < 1:
        return np.zeros(n)
    
    # Calculate pivot points
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Create arrays for each day (align with days)
    pivot_per_day = np.full(len(df_1d), np.nan)
    r1_per_day = np.full(len(df_1d), np.nan)
    s1_per_day = np.full(len(df_1d), np.nan)
    
    # Shift by one day: current day gets previous day's levels
    pivot_per_day[1:] = pivot
    r1_per_day[1:] = r1
    s1_per_day[1:] = s1
    
    # Align to 12h timeframe (only complete daily levels available)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_per_day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_per_day)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_per_day)
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R1, above daily EMA34, volume spike, volatility not extreme
            vol_condition = volume[i] > vol_ma[i] * 1.5
            vol_not_extreme = atr[i] < np.median(atr[max(0, i-50):i+1]) * 3  # Avoid volatility spikes
            
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_condition and 
                vol_not_extreme):
                signals[i] = 0.25
                position = 1
            # Short: price < S1, below daily EMA34, volume spike, volatility not extreme
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_condition and 
                  vol_not_extreme):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S1 or below EMA34 or volatility spike
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_34_aligned[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R1 or above EMA34 or volatility spike
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_34_aligned[i] or
                atr[i] > np.median(atr[max(0, i-50):i+1]) * 4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R1/S1 breakout with daily EMA34 trend filter, volume confirmation, and volatility filter.
# Uses previous day's Camarilla levels (R1/S1) as key support/resistance.
# Breakout above R1 with volume suggests bullish momentum; breakdown below S1 suggests bearish.
# Daily EMA(34) filter ensures we only trade in the direction of the daily trend.
# Volume confirmation ensures institutional participation.
# Volatility filter avoids whipsaws during extreme volatility spikes.
# Works in bull markets (buy breakouts above R1 in uptrend) and bear markets (sell breakdowns below S1 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~12-30 trades/year).