#!/usr/bin/env python3
"""
Experiment #1912: 12h Donchian(20) Breakout + 1d EMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture significant price moves when aligned with 1d EMA(50) trend and volume confirmation (>1.5x average). 
This strategy targets fewer, higher-quality trades by requiring confluence of price structure (breakout), trend filter (1d EMA), and volume spike. 
Designed to work in both bull and bear markets by following the dominant 1d trend while using 12h for precise entry timing. 
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1912_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    # Calculate rolling max/min for Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss and Donchian opposite touch ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            # Simplified ATR calculation using high-low range
            tr1 = high[i] - low[i]
            tr2 = np.abs(high[i] - close[i-1]) if i > 0 else tr1
            tr3 = np.abs(low[i] - close[i-1]) if i > 0 else tr1
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            # Use rolling average of TR for ATR(14)
            if i >= 14:
                atr_14 = pd.Series([tr if j==i else np.nan for j in range(i+1)]).rolling(window=14, min_periods=14).mean().iloc[-1]
            else:
                atr_14 = np.nan
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2 * ATR below entry
                if not np.isnan(atr_14) and price < entry_price - 2.0 * atr_14:
                    exit_signal = True
                # Exit if price touches Donchian lower channel (20-period low)
                elif price <= low_min[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2 * ATR above entry
                if not np.isnan(atr_14) and price > entry_price + 2.0 * atr_14:
                    exit_signal = True
                # Exit if price touches Donchian upper channel (20-period high)
                elif price >= high_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation: volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper channel AND 1d trend up
            if trend_1d_aligned[i] > 0 and price > high_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower channel AND 1d trend down
            elif trend_1d_aligned[i] < 0 and price < low_min[i]:
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