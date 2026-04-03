#!/usr/bin/env python3
"""
Experiment #154: 1h RSI(14) Extreme Reversal + 4h Volume Spike + 1d ADX(14) Trend Filter

HYPOTHESIS: RSI extremes (<30 or >70) on 1h combined with 4h volume spikes (>2.0x average) 
and 1d ADX(14) > 25 (trending market) capture high-probability reversals in both bull and bear 
markets. The 4h volume spike confirms institutional participation, while 1d ADX ensures we 
only trade in trending conditions where reversals are more meaningful. Uses discrete 
position sizing (0.20) and ATR-based stoploss (2.0x ATR) to manage risk. Targets 15-37 
trades/year by using strict entry conditions and session filter (08-20 UTC) to reduce 
fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_154_1h_rsi_extreme_4h_vol_1d_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume MA and ADX calculation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume MA(20) on 4h data
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate ADX(14) on 1d data
    df_1d = get_htf_data(prices, '1d')
    
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan)
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First TR is undefined
        
        # Directional Movement
        dm_plus = np.where((high_arr[1:] - high_arr[:-1]) > (low_arr[:-1] - low_arr[1:]), 
                           np.maximum(high_arr[1:] - high_arr[:-1], 0), 0)
        dm_minus = np.where((low_arr[:-1] - low_arr[1:]) > (high_arr[1:] - high_arr[:-1]), 
                            np.maximum(low_arr[:-1] - low_arr[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1h Indicators: RSI(14) ===
    def calculate_rsi(close_arr, period=14):
        if len(close_arr) < period + 1:
            return np.full_like(close_arr, np.nan)
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Prepend first value as NaN
        rsi = np.concatenate([[np.nan], rsi])
        return rsi
    
    rsi_14 = calculate_rsi(close, 14)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for stable indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(rsi_14[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- 1h RSI Extreme Conditions ---
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # --- 4h Volume Spike: Require volume > 2.0x average ---
        volume_spike = volume[i] > (2.0 * vol_ma_4h_aligned[i])
        
        # --- 1d ADX Trend Filter: Require ADX > 25 (trending market) ---
        strong_trend = adx_1d_aligned[i] > 25
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on RSI returning to neutral zone (40-60)
            if 40 <= rsi_14[i] <= 60:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: RSI oversold + volume spike + strong trend
        long_condition = rsi_oversold and volume_spike and strong_trend
        
        # Short: RSI overbought + volume spike + strong trend
        short_condition = rsi_overbought and volume_spike and strong_trend
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals