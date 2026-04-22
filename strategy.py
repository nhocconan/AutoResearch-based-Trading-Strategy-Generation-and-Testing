#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend context (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Load 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's pivot points
    prev_high = high_12h
    prev_low = low_12h
    prev_close = close_12h
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (high_12h - low_12h)
    s2 = pivot - (high_12h - low_12h)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation: 10-period average
    vol_avg_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema[i]) or np.isnan(atr_daily[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_avg_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA34 AND breaks above R2 with volume spike
            if (close[i] > weekly_ema[i] and close[i] > r2_aligned[i] and 
                volume[i] > 2.0 * vol_avg_10[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price below weekly EMA34 AND breaks below S2 with volume spike
            elif (close[i] < weekly_ema[i] and close[i] < s2_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_10[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses back to weekly EMA34
            if position == 1:
                # Exit long: Price closes below weekly EMA34
                if close[i] < weekly_ema[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Exit short: Price closes above weekly EMA34
                if close[i] > weekly_ema[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "12H_WeeklyEMA34_Trend_PivotR2S2_Volume"
timeframe = "12h"
leverage = 1.0