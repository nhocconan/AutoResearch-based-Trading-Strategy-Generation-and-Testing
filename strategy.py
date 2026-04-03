#!/usr/bin/env python3
"""
Experiment #1712: 12h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones. 
Price touching these levels with volume confirmation and 1d trend alignment provides high-probability 
mean-reversion entries in ranging markets and pullback entries in trending markets. 
12h timeframe reduces noise while capturing multi-day swings. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1712_12h_camarilla_pivot_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels from previous 1d bar
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4), S2 = C - ((H-L) * 1.1/6), S1 = C - ((H-L) * 1.1/12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    r2_1d = close_1d + (range_1d * 1.1 / 6.0)
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    s2_1d = close_1d - (range_1d * 1.1 / 6.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # 1d trend filter: EMA(21)
    ema_21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_21_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Align HTF arrays to 12h timeframe (with shift(1) for completed bars only)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
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
            # Mean reversion at Camarilla levels
            # In uptrend: look for longs at S1/S2, shorts at R1/R2
            # In downtrend: look for longs at S1/S2, shorts at R1/R2
            # Always fade extreme levels (S3/S4, R3/R4) with trend
            
            # Long conditions: price near support levels
            near_s1 = abs(price - s1_1d_aligned[i]) / price < 0.002  # within 0.2%
            near_s2 = abs(price - s2_1d_aligned[i]) / price < 0.002
            near_s3 = abs(price - s3_1d_aligned[i]) / price < 0.002
            near_s4 = abs(price - s4_1d_aligned[i]) / price < 0.002
            
            # Short conditions: price near resistance levels
            near_r1 = abs(price - r1_1d_aligned[i]) / price < 0.002
            near_r2 = abs(price - r2_1d_aligned[i]) / price < 0.002
            near_r3 = abs(price - r3_1d_aligned[i]) / price < 0.002
            near_r4 = abs(price - r4_1d_aligned[i]) / price < 0.002
            
            # Enter long at support levels with volume spike
            if (near_s1 or near_s2 or near_s3 or near_s4) and trend_1d_aligned[i] != 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Enter short at resistance levels with volume spike
            elif (near_r1 or near_r2 or near_r3 or near_r4) and trend_1d_aligned[i] != 0:
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