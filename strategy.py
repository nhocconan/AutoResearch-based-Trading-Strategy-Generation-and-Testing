#!/usr/bin/env python3
"""
Experiment #954: 1h Camarilla Pivot + Volume Spike + ATR Stoploss
HYPOTHESIS: Camarilla pivot levels (R4/S4) on 4h timeframe capture institutional breakout levels. 
Long when price breaks above R4 with volume spike (>1.5x avg volume) during active session (08-20 UTC). 
Short when price breaks below S4 with volume spike. Mean reversion at R3/S3 with reversal candle. 
Uses discrete position sizing (0.20) to limit drawdown. Target: 60-150 total trades over 4 years (15-37/year) on 1h.
Uses 4h for signal direction, 1h only for entry timing with session filter to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_954_1h_camarilla_pivot_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    # === HTF: 4h data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r4_4h = close_4h + (range_4h * 1.1 / 2.0)
    r3_4h = close_4h + (range_4h * 1.1 / 4.0)
    s3_4h = close_4h - (range_4h * 1.1 / 4.0)
    s4_4h = close_4h - (range_4h * 1.1 / 2.0)
    
    # Align Camarilla levels to 1h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for volume MA
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade during active hours (08-20 UTC) ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # --- Data Validity Check ---
        if (not in_session or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 24 bars (~1d on 1h) to avoid overtrading
            if bars_since_entry > 24:
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
            # Breakout continuation: price breaks above R4 OR below S4
            if price > r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Mean reversion: price at R3/S3 with reversal candle
            elif abs(price - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.1:
                # Near R3, look for bearish reversal (close < open)
                if close[i] < prices["open"].iloc[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif abs(price - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.1:
                # Near S3, look for bullish reversal (close > open)
                if close[i] > prices["open"].iloc[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals