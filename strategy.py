#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_VolumeRegime_ATRStop
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts filtered by daily EMA34 trend and volume regime (choppiness < 50 = trending).
Enter long when price breaks above 12h R1 with daily uptrend and trending regime.
Enter short when price breaks below 12h S1 with daily downtrend and trending regime.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite level break.
Designed for low trade frequency (target: 12-30 trades/year) to minimize fee drag.
Works in bull/bear via daily trend alignment and volume regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for pivots, 1d for trend/regime)
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
    
    # === Daily EMA34 for HTF trend filter ===
    close_1d = df_1d['close'].values
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
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_TR_14 / (HH_14 - LL_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.full_like(close_1d, 50.0, dtype=float)  # default to neutral
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_tr_14))
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === ATR (14-period) for stoploss (on 12h timeframe) ===
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
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime filter: chop < 50 indicates trending market
            trending_regime = chop_aligned[i] < 50.0
            
            # Volume confirmation: current volume > 20-period average (on 12h)
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_confirm = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 12h R1, daily uptrend, trending regime, volume spike
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_34_1d_aligned[i]
            
            # Short conditions: price < 12h S1, daily downtrend, trending regime, volume spike
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_34_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and trending_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and trending_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
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
            if price > entry_price + 2.5 * atr[i]:
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