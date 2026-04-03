#!/usr/bin/env python3
"""
Experiment #051: 6h Williams %R + 1d ADX Trend Filter + Volume Spike

HYPOTHESIS: Williams %R (14) on 6h timeframe identifies overbought/oversold conditions, 
filtered by 1d ADX (>25) to ensure trending markets and 12h volume spike (>1.8x average) 
to confirm institutional participation. Enters long when %R < -80 (oversold) in uptrend 
with volume, short when %R > -20 (overbought) in downtrend with volume. Uses discrete 
position sizing (0.25) and ATR-based stops (2.5x) to manage risk. Targets 12-37 trades/year 
on 6h timeframe to minimize fee drag while capturing high-probability mean reversals in 
trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.maximum(high_1d - low_1d, 
                       np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                                 abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 0.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, 
                   np.maximum(abs(high - np.roll(close, 1)), 
                             abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        trending_market = adx_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # --- Williams %R Levels ---
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral territory
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral territory
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold in uptrend with volume
        long_condition = oversold and trending_market and volume_spike
        
        # Short: Overbought in downtrend with volume
        short_condition = overbought and trending_market and volume_spike
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals