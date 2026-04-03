#!/usr/bin/env python3
"""
Experiment #045: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1d HMA trend direction and 
confirmed by 12h volume spike, captures strong momentum moves in both bull and bear markets. 
The 1d HMA ensures alignment with higher timeframe trend, reducing counter-trend trades. 
Volume confirmation ensures institutional participation. Targets 12-37 trades/year (50-150 total) 
on 12h timeframe to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian(20) channels on 12h
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        vol_12h = df_12h['volume'].values
        
        # Donchian channels
        donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Volume ratio (current vs 20-period average)
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        
        # Align to 12h timeframe
        donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
        donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        donch_high_20_aligned = np.full(n, np.nan)
        donch_low_20_aligned = np.full(n, np.nan)
        vol_ratio_12h_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA slope direction ---
        if i >= warmup + 1:
            hma_slope = hma_21_1d_aligned[i] - hma_21_1d_aligned[i-1]
            hma_uptrend = hma_slope > 0
            hma_downtrend = hma_slope < 0
        else:
            hma_uptrend = True  # Default to allow trading
            hma_downtrend = True
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
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
                # Take profit at Donchian low (trailing stop)
                if close[i] <= donch_low_20_aligned[i]:
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
                # Take profit at Donchian high (trailing stop)
                if close[i] >= donch_high_20_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and HMA uptrend
        long_condition = (
            close[i] > donch_high_20_aligned[i] and 
            volume_spike and 
            hma_uptrend
        )
        
        # Short: Price breaks below Donchian low with volume and HMA downtrend
        short_condition = (
            close[i] < donch_low_20_aligned[i] and 
            volume_spike and 
            hma_downtrend
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