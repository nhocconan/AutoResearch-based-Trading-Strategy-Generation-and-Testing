#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeRegime
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and chop regime filter (CHOP < 50) captures strong institutional moves while avoiding choppy markets. Uses ATR(14) stoploss (2.5) and discrete sizing (0.25). Targets 12-37 trades/year by requiring strict confluence of price level, daily trend, volume, and low-chop regime. Designed to work in both bull and bear markets via trend filter and regime avoidance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous completed 1d bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ATR(14) on 12h for stoploss
    # Since we don't have get_htf_data for 12h, we approximate using 4h data aligned to 12h
    # Alternative: calculate ATR directly on 12h bars by grouping, but we'll use 4h as proxy
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Align 4h ATR to 12h timeframe (each 12h bar = 3 4h bars)
    atr_12h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter on 1d (CHOP < 50 = strong trending regime)
    n_chop = 14
    tr_1d = []
    for i in range(1, len(high_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_sum = pd.Series(tr_1d).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_minus_min = pd.Series(high_1d - low_1d).rolling(window=n_chop, min_periods=n_chop).max().values
    chop_raw = 100 * np.log10(atr_sum / (n_chop * max_minus_min)) / np.log10(n_chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 34, 20, 14, 14)  # EMA34, vol MA, ATR, Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade in strong trending markets (CHOP < 50)
        in_strong_trend = chop_val < 50
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and in_strong_trend
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and in_strong_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks below S1 (exit long)
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks above R1 (exit short)
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeRegime"
timeframe = "12h"
leverage = 1.0