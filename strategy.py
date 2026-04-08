#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v13
# Hypothesis: Use daily pivot points as breakout levels (more reliable than fractals) with volume confirmation and daily EMA50 trend filter.
# Pivot points provide clear support/resistance levels that work in both bull/bear markets. Volume confirms breakout strength.
# Target: 15-30 trades/year by requiring pivot breakout + volume > 4x average + EMA50 trend alignment.
# Position size: 0.25 to manage drawdown in volatile markets like 2022.

name = "4h_fractal_breakout_1d_trend_volume_v13"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points and support/resistance levels."""
    n = len(high)
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get daily data for pivot points and trend filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate daily pivot points
    pivot_d, r1_d, r2_d, s1_d, s2_d = calculate_pivot_points(high_d, low_d, close_d)
    
    # Calculate daily EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 20-period average volume for 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate RSI for momentum filter (optional filter)
    rsi = calculate_rsi(close, 14)
    
    # Calculate ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track bars since entry for minimum holding period
    
    # Start from sufficient lookback
    start_idx = max(50, 20, 14)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Get aligned daily indicators for current 4h bar
        pivot_val = align_htf_to_ltf(prices, df_d, pivot_d)[i]
        r1_val = align_htf_to_ltf(prices, df_d, r1_d)[i]
        r2_val = align_htf_to_ltf(prices, df_d, r2_d)[i]
        s1_val = align_htf_to_ltf(prices, df_d, s1_d)[i]
        s2_val = align_htf_to_ltf(prices, df_d, s2_d)[i]
        ema50_val = align_htf_to_ltf(prices, df_d, ema50_d)[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        atr_ma_val = atr_ma[i]
        rsi_val = rsi[i]
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(r2_val) or 
            np.isnan(s1_val) or np.isnan(s2_val) or np.isnan(ema50_val) or
            np.isnan(vol_ma_val) or np.isnan(atr_val) or np.isnan(atr_ma_val) or
            volume[i] == 0 or np.isnan(rsi_val)):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volatility filter: current ATR < 1.5x 20-period average ATR (avoid choppy markets)
        vol_filter = atr_val < 1.5 * atr_ma_val
        
        # Volume breakout condition: current volume > 4.0x 20-period average
        vol_breakout = volume[i] > 4.0 * vol_ma_val
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema50_val
        downtrend = close[i] < ema50_val
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit if price breaks below pivot (support) OR minimum holding period met
            if (not np.isnan(pivot_val) and close[i] < pivot_val) or bars_since_entry >= 4:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit if price breaks above pivot (resistance) OR minimum holding period met
            if (not np.isnan(pivot_val) and close[i] > pivot_val) or bars_since_entry >= 4:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            bars_since_entry = 0
            # Breakout long above R1 (first resistance) with volume confirmation, uptrend
            if (not np.isnan(r1_val) and high[i] >= r1_val and 
                close[i] > r1_val and vol_breakout and uptrend):
                position = 1
                signals[i] = 0.25
            # Breakout short below S1 (first support) with volume confirmation, downtrend
            elif (not np.isnan(s1_val) and low[i] <= s1_val and 
                  close[i] < s1_val and vol_breakout and downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals