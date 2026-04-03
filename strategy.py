#!/usr/bin/env python3
"""
Experiment #1139: 6h Camarilla Pivot Fade/Breakout + Volume + 12h Trend
HYPOTHESIS: Camarilla pivot levels on 6h derived from 1d OHLC provide high-probability fade zones at R3/S3 and breakout continuation at R4/S4. 12h trend filter ensures alignment with higher timeframe momentum. Volume confirmation (>1.3x avg) filters low-quality signals. Designed to work in ranging markets (fades) and trending markets (breakouts). Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1139_6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(close_1d)
    
    # Calculate Camarilla levels for 1d timeframe
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align to 6h timeframe (shifted by 1 for completed 1d bar only)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Trend: price > EMA20 = uptrend, < = downtrend
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h[0:20] = 0  # warmup
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(trend_12h_aligned[i]) or
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Fade at R3/S3: price reaches extreme level and reverses
            # Short at R3 if price crosses below R3 after being above it
            # Long at S3 if price crosses above S3 after being below it
            # Breakout continuation at R4/S4: price breaks extreme level with trend
            
            # Check for R3 fade (short)
            if (price < r3_1d_aligned[i] and 
                close[i-1] >= r3_1d_aligned[i] and  # crossed below R3
                trend_12h_aligned[i] < 0):  # 12h downtrend alignment
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            
            # Check for S3 fade (long)
            elif (price > s3_1d_aligned[i] and 
                  close[i-1] <= s3_1d_aligned[i] and  # crossed above S3
                  trend_12h_aligned[i] > 0):  # 12h uptrend alignment
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            
            # Check for R4 breakout (long) - continuation
            elif (price > r4_1d_aligned[i] and 
                  close[i-1] <= r4_1d_aligned[i] and  # crossed above R4
                  trend_12h_aligned[i] > 0):  # 12h uptrend alignment
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            
            # Check for S4 breakout (short) - continuation
            elif (price < s4_1d_aligned[i] and 
                  close[i-1] >= s4_1d_aligned[i] and  # crossed below S4
                  trend_12h_aligned[i] < 0):  # 12h downtrend alignment
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