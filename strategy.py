#!/usr/bin/env python3
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
    
    # Load 1w data for weekly trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Load 1d data for daily pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    ranges = np.maximum(high_low, np.maximum(high_close, low_close))
    ranges[0] = high_low[0]  # First value
    tr = pd.Series(ranges).rolling(window=14, min_periods=14).mean().values
    atr_14 = align_htf_to_ltf(prices, df_1d, tr)
    
    # 6-period RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_20w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(r2[i]) or np.isnan(s2[i]) or np.isnan(atr_14[i]) or
            np.isnan(rsi_values[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA20 (uptrend) + breaks above R2 with volume + RSI > 50
            if (close[i] > ema_20w_aligned[i] and 
                close[i] > r2[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                rsi_values[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA20 (downtrend) + breaks below S2 with volume + RSI < 50
            elif (close[i] < ema_20w_aligned[i] and 
                  close[i] < s2[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  rsi_values[i] < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price closes below S1 OR RSI < 40
                if close[i] < s1[i] or rsi_values[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1 OR RSI > 60
                if close[i] > r1[i] or rsi_values[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyEMA20_DailyPivot_RSI_Volume"
timeframe = "6h"
leverage = 1.0