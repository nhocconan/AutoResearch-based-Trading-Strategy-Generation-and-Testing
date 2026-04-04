#!/usr/bin/env python3
"""
Experiment #4075: 6h Camarilla Pivot + Volume Spike + ATR Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as institutional support/resistance. 
Breakouts beyond R4/S4 with volume confirmation capture strong continuation moves, while reversals at R3/S3 
with volume exhaustion capture mean reversion in ranging markets. The 6h timeframe filters noise while 
capturing multi-day moves. Works in both bull/bear markets as pivots adapt to recent price action.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4075_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d OHLC for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla levels from previous 1d bar
        c_high = df_1d['high'].values
        c_low = df_1d['low'].values
        c_close = df_1d['close'].values
        
        # Camarilla formulas
        pivot = (c_high + c_low + c_close) / 3
        range_val = c_high - c_low
        
        # Resistance levels
        r3 = pivot + (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        # Support levels
        s3 = pivot - (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1)
        
        # Align to 6h timeframe (use previous day's levels)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # Volume MA lookback
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Fixed stoploss: 2.0 * ATR
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike filter (> 1.8x average) to ensure participation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Breakout beyond R4/S4 (strong continuation)
            breakout_long = price > r4_aligned[i]
            breakout_short = price < s4_aligned[i]
            
            # Reversal at R3/S3 (mean reversion from extreme levels)
            reversal_long = price < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]
            reversal_short = price > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]
            
            # Long conditions: breakout above R4 OR reversal from S3
            long_entry = breakout_long or reversal_long
            
            # Short conditions: breakout below S4 OR reversal from R3
            short_entry = breakout_short or reversal_short
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals