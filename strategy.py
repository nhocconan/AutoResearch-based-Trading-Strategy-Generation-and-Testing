#!/usr/bin/env python3
"""
Experiment #671: 6h Camarilla Pivot Reversal + 1d Volume Confirmation + ATR Stoploss
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as strong support/resistance.
Price rejecting at R3/S3 with volume confirmation indicates reversal. Break of R4/S4 indicates
continuation. Works in both bull/bear markets as pivots adapt to price action. Uses 6h timeframe 
to target 50-150 total trades over 4 years (12-37/year). Volume confirmation (>1.5x average) 
filters false signals. ATR-based stops (2.5x) manage risk. Discrete position sizing (0.25) 
minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_671_6h_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla Pivots and Volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    pivot_range = (high_1d - low_1d)
    camarilla_multiplier = 1.1 * 1.1 / 2  # 1.21/2 = 0.605
    camarilla_multiplier_half = camarilla_multiplier / 2  # 0.3025
    
    r4 = close_1d + camarilla_multiplier * pivot_range
    r3 = close_1d + camarilla_multiplier_half * pivot_range
    s3 = close_1d - camarilla_multiplier_half * pivot_range
    s4 = close_1d - camarilla_multiplier * pivot_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for pivots and ATR calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x 1d average) ---
        volume_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~3 days on 6h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Price rejects S3 (holds above S3) and breaks above R3 with volume
            # Short: Price rejects R3 (holds below R3) and breaks below S3 with volume
            if price > s3_aligned[i] and price < r3_aligned[i]:
                # Inside R3-S3 range - no entry
                signals[i] = 0.0
            elif price >= r3_aligned[i] and price < r4_aligned[i]:
                # In R3-R4 zone: potential reversal if rejects R3
                if low[i] <= r3_aligned[i] * 1.001 and close[i] > open[i]:  # Bullish rejection
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            elif price <= s3_aligned[i] and price > s4_aligned[i]:
                # In S3-S4 zone: potential reversal if rejects S3
                if high[i] >= s3_aligned[i] * 0.999 and close[i] < open[i]:  # Bearish rejection
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif price >= r4_aligned[i]:
                # Break above R4: continuation long
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price <= s4_aligned[i]:
                # Break below S4: continuation short
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals