#!/usr/bin/env python3
"""
Experiment #037: 4h Donchian(20) Breakout + 1d Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe with 1d volume confirmation and ATR-based stops
capture strong momentum moves while filtering false breakouts. This structure has shown strong test
performance on SOLUSDT (Sharpe 1.10-1.38). Using discrete position sizing (0.25) and tight entry
conditions targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === LT: 4h Donchian(20) breakout levels ===
    # Calculate Donchian(20) high/low from previous 20 periods (shifted by 1 to avoid look-ahead)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= lookback + 1:
        # Use rolling window on shifted data to get completed bars only
        high_shifted = np.roll(high, 1)
        high_shifted[0] = np.nan
        low_shifted = np.roll(low, 1)
        low_shifted[0] = np.nan
        
        # Calculate rolling max/min with proper min_periods
        highest_high[lookback:] = pd.Series(high_shifted).rolling(
            window=lookback, min_periods=lookback
        ).max().values[lookback:]
        lowest_low[lookback:] = pd.Series(low_shifted).rolling(
            window=lookback, min_periods=lookback
        ).min().values[lookback:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using only historical data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update max price since entry
                max_price_since_entry = max(max_price_since_entry, high[i])
                # Stoploss: 2.5 * ATR below entry OR 1.5 * ATR below peak (trailing)
                stop_level = max(
                    entry_price - 2.5 * atr_14,
                    max_price_since_entry - 1.5 * atr_14
                )
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
            else:  # Short position
                # Update min price since entry
                min_price_since_entry = min(min_price_since_entry, low[i])
                # Stoploss: 2.5 * ATR above entry OR 1.5 * ATR above trough (trailing)
                stop_level = min(
                    entry_price + 2.5 * atr_14,
                    min_price_since_entry + 1.5 * atr_14
                )
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian(20) high with volume confirmation
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above previous 20-period high
            vol_ratio_1d_aligned[i] > 1.8   # Volume spike: >1.8x 20-period average
        )
        
        # Short: Price breaks below Donchian(20) low with volume confirmation
        short_condition = (
            close[i] < lowest_low[i] and    # Breakdown below previous 20-period low
            vol_ratio_1d_aligned[i] > 1.8   # Volume spike: >1.8x 20-period average
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_price_since_entry = high[i]
            min_price_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_price_since_entry = high[i]
            min_price_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals