#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike confirmation, and chop regime filter. Uses discrete position sizing (0.25) to reduce fee drag. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year). Works in both bull and bear markets by following 1d trend direction while using Camarilla levels for precise entries. Chop regime filter avoids whipsaws in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # ATR for chop regime filter (14-period ATR on 4h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop regime filter: ATR(14) / ATR(50) < 0.7 = trending regime (good for breakouts)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    chop_regime = (atr / np.where(atr_50 == 0, np.nan, atr_50)) < 0.7
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need 1d EMA34 (34) + volume avg (20) + ATR (50 for chop filter)
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_regime[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_regime[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike, and chop regime
            # Long: price closes above R1 AND above EMA34 (1d uptrend) AND volume spike AND chop regime
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf and chop_ok
            # Short: price closes below S1 AND below EMA34 (1d downtrend) AND volume spike AND chop regime
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf and chop_ok
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions:
            # 1. Price touches S1 (opposite Camarilla level)
            # 2. 1d EMA34 turns bearish (price below EMA)
            exit_condition = (close_val < s1_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches R1 (opposite Camarilla level)
            # 2. 1d EMA34 turns bullish (price above EMA)
            exit_condition = (close_val > r1_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0