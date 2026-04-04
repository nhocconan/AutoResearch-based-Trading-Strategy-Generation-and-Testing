#!/usr/bin/env python3
"""
Experiment #3559: 6h Donchian Breakout + 12h Volume + 1d ATR Regime Filter
HYPOTHESIS: 6h Donchian(20) breakouts with 12h volume confirmation and 1d ATR regime filter capture medium-term momentum while avoiding choppy markets. Volume confirms breakout strength, ATR regime filter avoids false breakouts in low volatility environments. Position size 0.25. Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3559_6h_donchian20_12h_vol_1d_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    
    # Volume MA(10) on 12h for spike detection
    vol_ma_12h = pd.Series(vol_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = np.ones(len(vol_12h))
    vol_ratio_12h[10:] = vol_12h[10:] / vol_ma_12h[10:]
    
    # Align 12h volume ratio to 6h timeframe
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for ATR regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period MA ATR (volatility regime)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = np.ones(len(atr_1d))
    atr_ratio_1d[50:] = atr_1d[50:] / atr_ma_50[50:]
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: ATR(14) for stop loss ===
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_6h, 10, 50, 14) + 1  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average on 12h) for confirmation
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # Require high volatility regime (ATR ratio > 1.2) to avoid choppy markets
        high_vol_regime = atr_ratio_1d_aligned[i] > 1.2
        
        if volume_spike and high_vol_regime:
            # Long entry: price breaks above 6h Donchian high
            if price > highest_high_6h[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low
            elif price < lowest_low_6h[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals