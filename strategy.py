#!/usr/bin/env python3
name = "4h_PivotBreakout_MultiTF_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D and 1W data ONCE for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4H timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly trend filter: price above/below weekly EMA20
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 30-period average (more stringent)
    volume_s = pd.Series(volume)
    vol_ma30 = volume_s.rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > vol_ma30
    
    # Volatility filter: ATR(14) < 50-period MA of ATR (low volatility regime)
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # First TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    low_volatility = atr14 < atr_ma50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(weekly_trend_up[i]) or
            np.isnan(atr14[i]) or np.isnan(atr_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to weekly EMA20 (trend filter)
        price_above_weekly_ema = close[i] > weekly_trend_up[i]
        price_below_weekly_ema = close[i] < weekly_trend_up[i]
        
        if position == 0:
            # LONG: Break above R1 with volume, weekly uptrend, and low volatility
            if (close[i] > r1_4h[i] and price_above_weekly_ema and 
                volume_ok[i] and low_volatility[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume, weekly downtrend, and low volatility
            elif (close[i] < s1_4h[i] and price_below_weekly_ema and 
                  volume_ok[i] and low_volatility[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back to pivot or volatility increases
            if (close[i] < pivot_4h[i] or not low_volatility[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back to pivot or volatility increases
            if (close[i] > pivot_4h[i] or not low_volatility[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals