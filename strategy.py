#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop_v1
Hypothesis: 4h Camarilla R1/S1 breakout filtered by 1d EMA50 trend and choppiness regime (CHOP>61.8 = range, <38.2 = trend).
Enters only in trending regime (CHOP<38.2) with volume spike (>2x 20-period average) and 1d trend alignment.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
Target: 20-40 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend and CHOP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h OHLC for price action ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d OHLC for Camarilla pivots ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    r4_1d = df_1d_close + 1.5 * range_1d
    s4_1d = df_1d_close - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Choppiness Index (CHOP) on 1d: 14-period ===
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).sum().values  # SUM of TR over 14 periods
    high_max_1d = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    low_min_1d = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(atr_1d) * np.log10(14)
    chop_num = np.log10((high_max_1d - low_min_1d) / atr_1d)
    chop_1d = 100 * chop_num / chop_denom
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === ATR (14-period) for stoploss on 4h ===
    tr1_4h = pd.Series(high - low)
    tr2_4h = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_4h = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 20-period average on 4h ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) 
            or np.isnan(atr_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Regime filter: only trade in trending market (CHOP < 38.2)
            regime_filter = chop_1d_aligned[i] < 38.2
            # Volume filter: current volume > 2x 20-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > R1 (breakout), 1d uptrend, regime + volume filter
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 1d downtrend, regime + volume filter
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - ONLY enter on regime + volume filter + trend alignment
            if long_breakout and long_trend and regime_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and regime_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop_v1"
timeframe = "4h"
leverage = 1.0