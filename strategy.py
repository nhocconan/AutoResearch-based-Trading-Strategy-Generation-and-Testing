#!/usr/bin/env python3
"""
Experiment #108: 12h Donchian(20) breakout + 1w HMA trend + 1d volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant momentum moves. 
Filtering with 1w HMA(21) ensures alignment with weekly trend, reducing counter-trend trades. 
1d volume confirmation (>1.5x average) ensures institutional participation. 
ATR-based stoploss (2.5x) manages risk. Targets 12-37 trades/year on 12h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing high-probability trends 
in both bull and bear markets. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
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
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_middle = np.full(n, np.nan)
    
    if len(high) >= 20 and len(low) >= 20:
        # Donchian upper: max(high, 20)
        donchian_upper[20:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[20:]
        # Donchian lower: min(low, 20)
        donchian_lower[20:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[20:]
        # Donchian middle: (upper + lower) / 2
        donchian_middle = (donchian_upper + donchian_lower) / 2
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 1w HMA trend ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price crosses back below Donchian middle (trend weakening)
                if close[i] < donchian_middle[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price crosses back above Donchian middle (trend weakening)
                if close[i] > donchian_middle[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume and above weekly HMA
        long_condition = (
            close[i] > donchian_upper[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Price breaks below Donchian lower with volume and below weekly HMA
        short_condition = (
            close[i] < donchian_lower[i] and 
            volume_spike and 
            price_below_hma
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