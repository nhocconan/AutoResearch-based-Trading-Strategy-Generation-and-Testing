#!/usr/bin/env python3
"""
Experiment #4531: 6h Donchian(20) Breakout + 1d Volume Spike + 6h ATR Regime Filter
HYPOTHESIS: 6h Donchian(20) breakouts with volume confirmation (>2.0x average volume) 
and volatility regime filter (ATR(14) > ATR(50)) capture strong momentum moves while 
avoiding false breakouts in low-volume, low-volatility environments. This strategy 
targets 50-150 total trades over 4 years (12-37/year) with position size 0.25. 
Works in both bull and bear markets by only trading breakouts with volume and 
volatility expansion, which occurs during genuine market moves regardless of direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4531_6h_donchian20_1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume MA(20) for 1d
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.array([])
    
    # Align 1d volume MA to 6h timeframe
    if len(vol_ma_1d) > 0:
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) and ATR(50) for volatility regime ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, min_periods=50, adjust=False).mean().values
    vol_regime = np.ones(n)  # 1 = high vol regime, 0 = low vol
    vol_regime[50:] = (atr_14[50:] > atr_50[50:]).astype(float)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 50)  # Donchian, vol MA, ATR(50) warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Higher timeframe volume spike: current volume > 1.5x 1d average volume
        htf_vol_spike = volume[i] > (1.5 * vol_ma_1d_aligned[i])
        
        # Volatility regime filter: only trade in high volatility environment
        vol_filter = vol_regime[i] > 0.5
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout with volume and volatility confirmation
        long_entry = breakout_up and volume_confirm and htf_vol_spike and vol_filter
        
        # Short conditions: downward breakout with volume and volatility confirmation
        short_entry = breakout_down and volume_confirm and htf_vol_spike and vol_filter
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals