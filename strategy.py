#!/usr/bin/env python3
"""
Hypothesis: 6h weekly pivot breakout with volume confirmation and daily EMA(50) trend filter.
Long when price breaks above weekly R1 with volume > 2x average and close > daily EMA(50);
Short when price breaks below weekly S1 with volume > 2x average and close < daily EMA(50).
Exit on opposite weekly pivot touch or 2x ATR stop. Weekly pivots provide stronger institutional levels,
reducing false breaks and capturing longer-term trends. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets via upward breakouts and in bear via downward breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Load daily data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stop (14-period on 6h)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily volume aligned to 6h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above weekly R1 with volume surge and close > daily EMA50
            if (price_high > r1_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_20_aligned[i] and
                price_close > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly S1 with volume surge and close < daily EMA50
            elif (price_low < s1_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_20_aligned[i] and
                  price_close < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite weekly pivot touch or 2x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch weekly S1 OR price < entry - 2*ATR
                if price_low < s1_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use R1 as entry level for long
                    entry_level = r1_aligned[i-1] if i >= 1 else r1_aligned[0]
                    if price_close < entry_level - 2.0 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch weekly R1 OR price > entry + 2*ATR
                if price_high > r1_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use S1 as entry level for short
                    entry_level = s1_aligned[i-1] if i >= 1 else s1_aligned[0]
                    if price_close > entry_level + 2.0 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_1dEMA50_Trend_Volume2x"
timeframe = "6h"
leverage = 1.0