#!/usr/bin/env python3
"""
Experiment #751: 6h Camarilla Pivot + Volume Spike + ADX Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with volume confirmation (>1.5x average) and ADX regime filter (ADX>25 for breakout, 
ADX<20 for mean reversion) captures institutional activity with proper HTF alignment. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear markets: 
mean reversion in range (ADX<20) at R3/S3, breakout continuation in trend (ADX>25) at R4/S4.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_751_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: P = (H+L+C)/3, Range = H-L
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    # Resistance levels: R3 = C + Range*1.1/2, R4 = C + Range*1.1
    # Support levels: S3 = C - Range*1.1/2, S4 = C - Range*1.1
    r3 = close_1d + camarilla_range * 1.1 / 2.0
    r4 = close_1d + camarilla_range * 1.1
    s3 = close_1d - camarilla_range * 1.1 / 2.0
    s4 = close_1d - camarilla_range * 1.1
    
    # Align Camarilla levels to 6h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            dm_plus[i] = high_diff
        else:
            dm_plus[i] = 0
        if low_diff > high_diff and low_diff > 0:
            dm_minus[i] = low_diff
        else:
            dm_minus[i] = 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = np.zeros(n)
    dx[14:] = 100 * np.abs(di_plus[14:] - di_minus[14:]) / (di_plus[14:] + di_minus[14:])
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 14)  # sufficient for volume MA and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (using 1.5*ATR for tighter stops) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR for stoploss
            atr_val = tr_ma[i]  # Using TR MA as proxy for ATR
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 1.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 1.5 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 4 bars (~24h on 6h) to avoid overtrading
            if bars_since_entry > 4:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Regime filter: ADX > 25 = trending (breakout), ADX < 20 = ranging (mean reversion)
            if adx[i] > 25:
                # Trending regime: breakout continuation at R4/S4
                if high[i] > r4_aligned[i]:
                    # Breakout above R4 -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif low[i] < s4_aligned[i]:
                    # Breakdown below S4 -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif adx[i] < 20:
                # Ranging regime: mean reversion at R3/S3
                if low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]:
                    # Bounce off S3 -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]:
                    # Rejection at R3 -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # ADX between 20-25: transition regime, no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals