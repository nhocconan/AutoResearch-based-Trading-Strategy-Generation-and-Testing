#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V1
Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1/S1) for breakout entries,
filtered by 1d EMA34 trend and 12h choppiness regime (CHOP > 50 = range, < 50 = trend).
Enter long when price breaks above 1d R1 with 1d uptrend and trending market (CHOP < 50).
Enter short when price breaks below 1d S1 with 1d downtrend and trending market.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite level break.
Target: 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
Works in bull/bear via HTF trend alignment and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align to 12h timeframe (use previous completed daily bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA34 for HTF trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Choppiness Index (CHOP) - 14 period
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div by zero
    
    # ATR (14-period) for stoploss
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(chop[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long conditions: price > 1d R1, 1d uptrend, trending market (CHOP < 50)
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_34_1d_aligned[i]
            long_regime = chop[i] < 50
            
            # Short conditions: price < 1d S1, 1d downtrend, trending market
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_34_1d_aligned[i]
            short_regime = chop[i] < 50
            
            # Entry logic
            if long_breakout and long_trend and long_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1d S1 (support broken)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1d R1 (resistance broken)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0