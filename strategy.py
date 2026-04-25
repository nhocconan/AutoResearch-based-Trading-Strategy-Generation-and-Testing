#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d trend filter (price > EMA50 for longs, < EMA50 for shorts) and volume spike confirmation (>1.5x 20-period average) captures institutional breakout moves in both bull and bear markets. Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price closes against position by 1.5x ATR). Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation - primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (R1, S1) using previous day's OHLC
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_12h - low_12h
    r1 = close_12h + 1.1 * camarilla_range / 12
    s1 = close_12h - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe (already aligned via get_htf_data)
    # Need to shift by 1 bar to avoid look-ahead (use previous bar's levels)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_shifted)
    
    # Get 1d data for trend filter (EMA50) and volume average - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate ATR for stoploss (using 12h data)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(60, 50)  # EMA50 needs 50, volume avg needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_12h_val = close_12h[i] if i < len(close_12h) else close_12h[-1]
        ema_50_val = ema_50_aligned[i]
        vol_avg_20_val = vol_avg_20_aligned[i]
        volume_12h_val = volume[i] if i < len(volume) else volume[-1]
        atr_val = atr_aligned[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = volume_12h_val > 1.5 * vol_avg_20_val
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume confirmation
            long_breakout = close_12h_val > r1_val
            short_breakout = close_12h_val < s1_val
            
            # Trend filter: price > EMA50 for longs, < EMA50 for shorts
            long_trend = close_12h_val > ema_50_val
            short_trend = close_12h_val < ema_50_val
            
            long_signal = long_breakout and long_trend and volume_spike
            short_signal = short_breakout and short_trend and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_12h_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_12h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # ATR-based stoploss: exit if price closes below entry - 1.5*ATR
            if close_12h_val < entry_price - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # ATR-based stoploss: exit if price closes above entry + 1.5*ATR
            if close_12h_val > entry_price + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0