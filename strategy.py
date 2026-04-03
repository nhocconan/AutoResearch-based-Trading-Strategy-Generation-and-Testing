#!/usr/bin/env python3
"""
Experiment #211: 6h Elder Ray + 1d ADX Trend Filter
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h captures momentum impulses, while 1d ADX > 25 filters for strong trending regimes. Long when Bull Power > 0 and Bear Power < 0 with ADX > 25; Short when Bear Power < 0 and Bull Power > 0 with ADX > 25. This combination works in bull markets via sustained Bull Power > 0 and in bear markets via sustained Bear Power < 0, with ADX preventing entries in choppy/range-bound conditions. Volume confirmation (>1.5x average) ensures breakout validity. ATR stoploss (2.5x) manages risk. Discrete position sizing (0.25) balances return and fee drag. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_211_6h_elder_ray_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) for 1d
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
        
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    # Calculate ADX for 1d
    adx_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        adx_values = calculate_adx(
            df_1d['high'].values,
            df_1d['low'].values,
            df_1d['close'].values
        )
        adx_1d[:len(adx_values)] = adx_values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume > 1.5x average ---
        volume_ok = vol_ratio[i] > 1.5
        
        # --- Trend Filter: Require ADX > 25 (strong trend) ---
        strong_trend = adx_1d_aligned[i] > 25
        
        # --- Elder Ray Conditions ---
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
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
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume confirmation + strong trend + Elder Ray alignment
        if volume_ok and strong_trend:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum)
            if bull_power_pos and bear_power_neg:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power < 0 AND Bull Power > 0 (bearish momentum)
            elif bear_power_neg and bull_power_pos:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals