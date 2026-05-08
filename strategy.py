#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d RSI filter and volume spike
# KAMA adapts to market noise - slow in ranging markets, fast in trends.
# We go long when KAMA slope > 0 and RSI(14) < 30 (oversold bounce in uptrend)
# Short when KAMA slope < 0 and RSI(14) > 70 (overbought rejection in downtrend)
# Volume spike confirms institutional participation.
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_KAMA_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(er_period))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
    sc = sc.fillna((2/(slow_sc+1))**2)  # fill first values with slow SC
    kama_vals = np.full_like(close, np.nan, dtype=float)
    kama_vals[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama_vals[i] = kama_vals[i-1] + sc.iloc[i] * (close[i] - kama_vals[i-1])
        else:
            kama_vals[i] = kama_vals[i-1]
    return kama_vals

def rsi(close, period=14):
    """Relative Strength Index"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50).values  # neutral RSI when undefined

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for overbought/oversold conditions
    close_1d = df_1d['close'].values
    rsi14_1d = rsi(close_1d, 14)
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # KAMA on 4h data
    kama_vals = kama(close, 10, 2, 30)
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama_vals, prepend=kama_vals[0])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi14_1d_aligned[i]) or np.isnan(kama_vals[i]) or 
            np.isnan(kama_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi14_1d_aligned[i]
        kama_slope_val = kama_slope[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: KAMA rising (uptrend) + RSI oversold + volume spike
            if (kama_slope_val > 0 and 
                rsi_val < 30 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling (downtrend) + RSI overbought + volume spike
            elif (kama_slope_val < 0 and 
                  rsi_val > 70 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA slope turns negative OR RSI becomes overbought
            if kama_slope_val <= 0 or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA slope turns positive OR RSI becomes oversold
            if kama_slope_val >= 0 or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals