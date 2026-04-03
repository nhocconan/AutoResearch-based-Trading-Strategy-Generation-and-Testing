#!/usr/bin/env python3
"""
Experiment #056: 12h Donchian Breakout + 1d Volume Spike + ATR Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by ATR-based volatility regime, captures strong trending moves while avoiding 
choppy markets. The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to 
minimize fee drag. Volume spike ensures institutional participation, ATR filter adapts to 
market volatility, and Donchian breakouts provide clear entry/exit levels. Designed to work 
in both bull and bear markets by trading breakouts in direction of prevailing trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and trend (Call ONCE before loop) ===
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
    
    # Calculate 1d EMA(50) for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            window_high = high[i-19:i+1]  # 20 periods including current
            window_low = low[i-19:i+1]
            highest_20[i] = np.max(window_high)
            lowest_20[i] = np.min(window_low)
            donchian_mid[i] = (highest_20[i] + lowest_20[i]) / 2
        else:
            highest_20[i] = np.nan
            lowest_20[i] = np.nan
            donchian_mid[i] = np.nan
    
    # Calculate ATR(14) on 12h for stoploss and volatility filter
    atr_14 = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volatility Regime Filter: Only trade when ATR > 20-period ATR mean (avoid choppy low-vol) ---
        if i >= 34:  # Need 20 + 14 for ATR mean calculation
            atr_ma_20 = np.mean(atr_14[i-19:i+1])
            volatility_filter = atr_14[i] > atr_ma_20 * 0.8  # Allow 20% below mean to avoid whipsaw
        else:
            volatility_filter = True  # Default to true during warmup
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss and Donchian middle reversion) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reverts below Donchian middle (trend weakening)
                if close[i] < donchian_mid[i]:
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
                # Exit if price reverts above Donchian middle (trend weakening)
                if close[i] > donchian_mid[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume spike and volatility filter
        long_condition = (
            close[i] > highest_20[i] and 
            volume_spike and 
            volatility_filter
        )
        
        # Short: Price breaks below Donchian lower with volume spike and volatility filter
        short_condition = (
            close[i] < lowest_20[i] and 
            volume_spike and 
            volatility_filter
        )
        
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