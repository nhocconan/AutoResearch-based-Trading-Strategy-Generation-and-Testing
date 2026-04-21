#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter
Hypothesis: 4h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA34 trend and choppiness regime.
Enter long when price breaks above daily R1 with daily uptrend and low chop (trending market).
Enter short when price breaks below daily S1 with daily downtrend and low chop.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite level break.
Designed for moderate trade frequency (target: 25-40 trades/year) to balance edge and fees.
Works in bull/bear via daily trend alignment and chop filter as regime detector.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for pivots and trend, 1d for chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align to daily timeframe (use previous completed daily bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Daily EMA34 for HTF trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(sum(ATR)/log10(N)) / log10(N)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    n_periods = 14
    chop = 100 * np.log10(sum_atr_14 / np.log10(n_periods)) / np.log10(n_periods)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === ATR (14-period) for 4h stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime filter: low chop (< 38.2) indicates trending market
            trending_regime = chop_aligned[i] < 38.2
            
            # Long conditions: price > daily R1, daily uptrend, trending regime
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_34_1d_aligned[i]
            
            # Short conditions: price < daily S1, daily downtrend, trending regime
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_34_1d_aligned[i]
            
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
            # Trailing exit: price closes below daily S1 (support broken)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above daily R1 (resistance broken)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0