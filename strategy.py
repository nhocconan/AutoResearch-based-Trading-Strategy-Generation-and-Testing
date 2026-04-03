#!/usr/bin/env python3
"""
Experiment #242: 12h Camarilla Pivot Breakout + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Trading Camarilla pivot level breakouts (H3/L3) on 12h timeframe with volume confirmation and 1d choppiness regime filter captures institutional breakout moves while avoiding false signals in ranging markets. The 12h timeframe minimizes fee drag, Camarilla levels provide mathematically derived support/resistance, volume confirms participation, and chop filter ensures trending conditions. Targets 12-37 trades/year (50-150 total over 4 years) for statistical validity and low fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on daily data
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Daily range
        daily_range = high_1d - low_1d
        
        # Pivot point (standard)
        pivot = (high_1d + low_1d + close_1d) / 3.0
        
        # Camarilla levels (based on daily range)
        # Resistance levels
        r4 = close_1d + daily_range * 1.500
        r3 = close_1d + daily_range * 1.250
        r2 = close_1d + daily_range * 1.166
        r1 = close_1d + daily_range * 1.083
        
        # Support levels
        s1 = close_1d - daily_range * 1.083
        s2 = close_1d - daily_range * 1.166
        s3 = close_1d - daily_range * 1.250
        s4 = close_1d - daily_range * 1.500
        
        # We'll use H3 (R3) and L3 (S3) for breakouts
        camarilla_h3 = r3
        camarilla_l3 = s3
        
        # Align to 12h timeframe (completed daily bars only)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for choppiness regime (Call ONCE before loop) ===
    # Calculate Choppiness Index(14) on 1d data
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1d = np.full(len(close_1d), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Volume spike detection (volume > 2.0 * 20-period average)
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (vol_ma_20 * 2.0)
    else:
        volume_spike = np.zeros(n, dtype=bool)
        vol_ma_20 = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid choppy markets (Choppiness > 61.8 = ranging) ---
        # Only trade when market is trending (Choppiness < 61.8) 
        if chop_1d_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        if not volume_spike[i]:
            signals[i] = 0.0
            continue
        
        # --- Camarilla Breakout Logic ---
        # Long breakout: price closes above H3 resistance
        long_breakout = close[i] > camarilla_h3_aligned[i]
        
        # Short breakout: price closes below L3 support
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if high[i] > close[entry_bar] + 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if low[i] < close[entry_bar] - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above H3 with volume spike
        if long_breakout:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Breakout below L3 with volume spike
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals