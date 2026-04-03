#!/usr/bin/env python3
"""
Experiment #952: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian(20) breakouts capture significant momentum shifts. 
Filtering by 1d EMA50 ensures we only trade in the direction of the higher timeframe trend, 
reducing whipsaws in ranging/bear markets. Volume confirmation (>1.5x average) adds conviction.
Discrete position sizing (0.25) limits drawdown. Target: 75-150 total trades over 4 years (19-37/year) on 12h timeframe.
Works in bull markets via breakout continuation and in bear markets via trend-filtered mean reversion at channel mid-point.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_952_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 and Donchian channel calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate Donchian(20) on 1d: upper = max(high,20), lower = min(low,20)
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_upper_1d = high_roll
    donch_lower_1d = low_roll
    donch_middle_1d = (donch_upper_1d + donch_lower_1d) / 2.0
    
    # Align 1d indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    donch_middle_aligned = align_htf_to_ltf(prices, df_1d, donch_middle_1d)
    
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
    
    warmup = max(50, 20, 20)  # EMA50, Donchian20, VolMA20
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_aligned[i]) or np.isnan(donch_upper_aligned[i]) or
            np.isnan(donch_lower_aligned[i]) or np.isnan(donch_middle_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 12h to avoid noise)
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
            
            # Optional: time-based exit after 8 bars (~4d on 12h) to avoid overtrading
            if bars_since_entry > 8:
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
            # Determine trend direction from 1d EMA50
            uptrend = price > ema50_aligned[i]
            downtrend = price < ema50_aligned[i]
            
            # Breakout continuation: trade in direction of trend
            if uptrend and price > donch_upper_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif downtrend and price < donch_lower_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Mean reversion: trade against extreme moves when price reverts to channel middle
            elif not uptrend and not downtrend:  # ranging market (price near EMA50)
                if price > donch_upper_aligned[i] * 0.995 and price < donch_upper_aligned[i] * 1.005:
                    # Near upper channel, look for bearish reversal
                    if close[i] < prices["open"].iloc[i]:
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                elif price < donch_lower_aligned[i] * 1.005 and price > donch_lower_aligned[i] * 0.995:
                    # Near lower channel, look for bullish reversal
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