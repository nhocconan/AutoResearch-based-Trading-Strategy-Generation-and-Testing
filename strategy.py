#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_VolumeRegime_ATRStop
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA50 trend and volume regime (choppiness < 50) to avoid whipsaw in ranging markets.
Enter long when price breaks above 12h R1 with 1d uptrend and low chop (trending regime).
Enter short when price breaks below 12h S1 with 1d downtrend and low chop.
Exit on ATR(20) trailing stop (2.0*ATR) or opposite level break.
Designed for low trade frequency (~20-40 trades/year) to minimize fee drag on 12h timeframe.
Works in bull/bear via 1d trend alignment and chop regime filter to avoid false breakouts in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for pivots, 1d for trend/chop)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    r1_12h = close_12h + camarilla_range
    s1_12h = close_12h - camarilla_range
    
    # Align to 12h timeframe (use previous completed 12h bar)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 1d EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum().values  # Sum of TR
    
    # Choppiness Index: 100 * log10(sum(TR) / (ATR * sqrt(N))) / log10(N)
    n_period = 14
    atr_avg = tr.rolling(window=n_period, min_periods=n_period).mean().values
    chop = 100 * np.log10(atr_1d / (atr_avg * np.sqrt(n_period))) / np.log10(n_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === ATR (20-period) for stoploss on 12h timeframe ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume regime: chop < 50 indicates trending market (avoid ranging)
            trending_regime = chop_aligned[i] < 50.0
            
            # Long conditions: price > 12h R1, 1d uptrend, trending regime
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < 12h S1, 1d downtrend, trending regime
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and trending_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and trending_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 12h S1 (support broken)
            elif price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 12h R1 (resistance broken)
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dTrend_VolumeRegime_ATRStop"
timeframe = "12h"
leverage = 1.0