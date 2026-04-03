#!/usr/bin/env python3
"""
Experiment #1948: 12h Donchian(20) breakout + 1w EMA trend + volume confirmation
HYPOTHESIS: 12h timeframe captures medium-term swings while avoiding lower TF noise. 
Donchian(20) breakouts with 1w EMA trend filter and volume confirmation provide 
high-probability entries in both bull and bear markets. Weekly EMA establishes 
the primary trend direction, reducing false breakouts. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1948_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian channels: 20-period high/low
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (2*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                # Simple ATR calculation using rolling mean
                atr = np.mean([tr] + [0]*13)  # placeholder for first bar
                if i >= 27:  # enough for 14-period ATR
                    atr_vals = []
                    for j in range(i-13, i+1):
                        tr1_j = high[j] - low[j]
                        tr2_j = abs(high[j] - close[j-1]) if j > 0 else tr1_j
                        tr3_j = abs(low[j] - close[j-1]) if j > 0 else tr1_j
                        tr_j = np.maximum(tr1_j, np.maximum(tr2_j, tr3_j))
                        atr_vals.append(tr_j)
                    atr = np.mean(atr_vals)
                else:
                    atr = np.mean([high[k] - low[k] for k in range(max(0, i-13), i+1)])
                
                # Stoploss: 2*ATR against position
                if position_side > 0:  # Long
                    if price < entry_price - 2.0 * atr:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price > entry_price + 2.0 * atr:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:
                # Not enough data for ATR yet, use fixed percentage
                if position_side > 0:  # Long
                    if price < entry_price * 0.97:  # 3% stoploss
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Short
                    if price > entry_price * 1.03:  # 3% stoploss
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 1w trend up
            if trend_1w_aligned[i] > 0 and price > high_ma[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 1w trend down
            elif trend_1w_aligned[i] < 0 and price < low_ma[i]:
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