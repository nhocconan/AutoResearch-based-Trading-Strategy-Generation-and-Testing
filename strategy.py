#!/usr/bin/env python3
"""
Experiment #019: 6h Williams %R + 12h ADX Trend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h, while 12h ADX(14) > 25 filters for trending regimes. 
In strong trends (ADX > 25), we fade extreme Williams %R readings (< -80 for long, > -20 for short) with volume confirmation (>1.3x average). 
This captures mean reversion within trends, which works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. 
Uses ATR-based stoploss (2.5x) and minimum 4-bar holding period. Target: 80-120 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_019_6h_williamsr_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values.astype(np.float64)
    low_12h = df_12h['low'].values.astype(np.float64)
    close_12h = df_12h['close'].values.astype(np.float64)
    
    # Calculate 12h ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        atr[tr == 0] = 1e-10  # Avoid division by zero
        
        di_plus = 100 * pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        di_minus = 100 * pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for 12h ADX and 6h indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Trend Filter: Require ADX > 25 for trending regime ---
        is_trending = adx_12h_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume > 1.3x average ---
        volume_confirm = vol_ratio[i] > 1.3
        
        # --- Williams %R Extreme Conditions ---
        williams_oversold = williams_r[i] < -80  # Extreme oversold
        williams_overbought = williams_r[i] > -20  # Extreme overbought
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on Williams %R returning to neutral territory
            if position_side > 0 and williams_r[i] > -50:  # Long exit when no longer oversold
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and williams_r[i] < -50:  # Short exit when no longer overbought
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade in trending regimes with volume confirmation
        if is_trending and volume_confirm:
            # Long: Williams %R extremely oversold in uptrend
            if williams_oversold:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Williams %R extremely overbought in downtrend
            elif williams_overbought:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging or low volume conditions, do not trade
            signals[i] = 0.0
    
    return signals