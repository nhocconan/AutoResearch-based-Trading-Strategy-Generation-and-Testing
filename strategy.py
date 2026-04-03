#!/usr/bin/env python3
"""
Experiment #119: 6h Camarilla Pivot Fade/Breakout + 12h Volume Spike + ADX Regime Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion fade, R4/S4 for breakout continuation) 
filtered by 12h volume confirmation (>2.0x average) and ADX regime filter (ADX>25 for breakout, 
ADX<20 for fade) capture high-probability mean reversion and momentum moves. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing 
(0.25) to minimize fee drag. Works in bull markets (breakouts at R4/S4 with volume) and bear 
markets (fades at R3/S3 during ranging conditions). Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_119_6h_camarilla_12h_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume and ADX regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume MA(20) on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        """Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF indicators to LTF
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from 1d data
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla levels: 
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe (each 6h bar gets previous day's levels)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Ensure enough data for HTF indicators and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = volume[i] > (2.0 * vol_ma_12h_aligned[i])
        
        # --- 12h ADX Regime Filter ---
        adx_high = adx_12h_aligned[i] > 25   # Trending regime (breakout)
        adx_low = adx_12h_aligned[i] < 20    # Ranging regime (mean reversion)
        
        # --- Camarilla Breakout/Fade Conditions ---
        # Breakout long: price > R4 with volume spike and ADX>25
        breakout_long = (close[i] > camarilla_r4_6h[i]) and volume_spike and adx_high
        
        # Breakout short: price < S4 with volume spike and ADX>25
        breakout_short = (close[i] < camarilla_s4_6h[i]) and volume_spike and adx_high
        
        # Fade long: price < S3 with volume spike and ADX<20 (mean reversion long)
        fade_long = (close[i] < camarilla_s3_6h[i]) and volume_spike and adx_low
        
        # Fade short: price > R3 with volume spike and ADX<20 (mean reversion short)
        fade_short = (close[i] > camarilla_r3_6h[i]) and volume_spike and adx_low
        
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
            
            # Exit conditions: opposite signal or Donchian-like middle reversion
            # For breakout positions, exit on reversion to midpoint
            # For fade positions, exit on reversion to pivot level (using R3/S3 as reference)
            pivot_mid = (camarilla_r3_6h[i] + camarilla_s3_6h[i]) / 2
            
            if position_side > 0:  # Long
                # Exit long on price below pivot mid (for fade) or strong reversal
                if close[i] < pivot_mid:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Exit short on price above pivot mid
                if close[i] > pivot_mid:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Priority: Breakout signals first (stronger momentum), then fade signals
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        elif fade_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif fade_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals