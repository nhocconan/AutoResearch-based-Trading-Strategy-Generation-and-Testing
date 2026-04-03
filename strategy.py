#!/usr/bin/env python3
"""
Experiment #179: 6h Volume Spike Donchian Breakout + 12h ADX Trend Filter + HTF Regime
HYPOTHESIS: Combining 6h Donchian(20) breakouts with volume confirmation (>2x avg volume) and 12h ADX trend filter (ADX>25) captures strong momentum moves. Uses 1d HTF regime (price > EMA50 for longs, < EMA50 for shorts) to align with higher timeframe direction. Works in both bull and bear markets by requiring volume confirmation and trend alignment. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_179_6h_volume_spike_donchian_12h_adx_1d_regime_v1"
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
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First bar
        
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        tr_ema = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_dm_ema = pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
        minus_dm_ema = pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values
        
        plus_di = 100 * plus_dm_ema / tr_ema
        minus_di = 100 * minus_dm_ema / tr_ema
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        
        # Handle division by zero
        adx = np.where((plus_di + minus_di) == 0, 0, adx)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_strong = adx_12h > 25  # Strong trend
    
    # Align ADX to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    
    # === HTF: 1d data for regime filter (price vs EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    regime_long = close_1d > ema50_1d   # Bullish regime
    regime_short = close_1d < ema50_1d  # Bearish regime
    
    # Align regime to 6h timeframe
    regime_long_aligned = align_htf_to_ltf(prices, df_1d, regime_long)
    regime_short_aligned = align_htf_to_ltf(prices, df_1d, regime_short)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume Spike Detection (>2x average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.zeros(n)
    vol_spike[20:] = volume[20:] > (2.0 * vol_ma_20[20:])
    vol_spike[:20] = False
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(regime_long_aligned[i]) or np.isnan(regime_short_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
        # Long: Price breaks above Donchian upper + volume spike + 12h ADX strong + 1d bullish regime
        if (price > donch_upper[i] and vol_spike[i] and 
            adx_strong_aligned[i] and regime_long_aligned[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower + volume spike + 12h ADX strong + 1d bearish regime
        elif (price < donch_lower[i] and vol_spike[i] and 
              adx_strong_aligned[i] and regime_short_aligned[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals