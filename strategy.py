#!/usr/bin/env python3
"""
Experiment #449: 4h Donchian(20) breakout + 1d HMA trend + 1w volume confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture strong momentum moves, 
filtered by 1d Hull Moving Average trend direction and 1w volume spike confirmation. 
This combines price structure (Donchian), trend filtering (HMA), and participation 
validation (volume) to work in both bull and bear markets. Targets 25-50 trades/year 
on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing 
high-probability breakouts with institutional volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_1w_vol_v1"
timeframe = "4h"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = len(close_1d) // 2
        sqrt_len = int(np.sqrt(len(close_1d)))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        # Pad arrays for convolution
        wma_half = np.full(len(close_1d), np.nan)
        wma_full = np.full(len(close_1d), np.nan)
        
        if half_len > 0 and len(close_1d) >= half_len:
            wma_half_vals = wma(close_1d, half_len)
            wma_half[half_len-1:half_len-1+len(wma_half_vals)] = wma_half_vals
        
        if len(close_1d) >= len(close_1d):
            wma_full_vals = wma(close_1d, len(close_1d))
            wma_full[len(close_1d)-1:len(close_1d)-1+len(wma_full_vals)] = wma_full_vals
            
        # Simplified HMA calculation using EMA approximation for stability
        # HMA ~= EMA(2*EMA(n/2) - EMA(n)) with period sqrt(n)
        ema_half = pd.Series(close_1d).ewm(span=half_len, min_periods=half_len, adjust=False).mean().values
        ema_full = pd.Series(close_1d).ewm(span=len(close_1d), min_periods=len(close_1d), adjust=False).mean().values
        hma_raw = 2 * ema_half - ema_full
        hma_21 = pd.Series(hma_raw).ewm(span=sqrt_len, min_periods=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate Donchian Channel (20) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    if n >= 20:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1] if i > 0 else False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.8
        
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
                # Take profit at Donchian midpoint or opposite band
                if close[i] >= donchian_high[i] or close[i] <= donchian_low[i]:
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
                # Take profit at Donchian midpoint or opposite band
                if close[i] >= donchian_high[i] or close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High with HMA uptrend and volume
        long_condition = (
            close[i] > donchian_high[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian Low with HMA downtrend and volume
        short_condition = (
            close[i] < donchian_low[i] and 
            hma_falling and 
            volume_spike
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