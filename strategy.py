#!/usr/bin/env python3
"""
Experiment #263: 4h Donchian(20) Breakout + 12h/1d Regime Filter + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends. 
Regime filter from 12h/1d (ADX > 25 = trend, ADX < 20 = range) adapts strategy: 
- In trend (ADX>25): breakout continuation (long on upper break, short on lower break)
- In range (ADX<20): mean reversion at Donchian edges (short upper break, long lower break)
Volume confirmation (>2.0x average) ensures strong participation. 
ATR stoploss (2.5x) limits drawdown. Discrete sizing 0.30 balances return and fees.
Target: 75-200 total trades over 4 years (19-50/year). Works in bull via trend continuation 
and in bear via mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_263_4h_donchian20_12h_1d_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    # ADX calculation on 12h
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), 
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    dm_plus_12h = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                           np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus_12h = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                            np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus_12h[0] = 0
    dm_minus_12h[0] = 0
    tr14_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus14_12h = pd.Series(dm_plus_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus14_12h = pd.Series(dm_minus_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    di_plus_12h = 100 * dm_plus14_12h / (tr14_12h + 1e-10)
    di_minus_12h = 100 * dm_minus14_12h / (tr14_12h + 1e-10)
    dx_12h = 100 * np.abs(di_plus_12h - di_minus_12h) / (di_plus_12h + di_minus_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === HTF: 1d data for Donchian channel reference (optional) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 4h Indicators: Donchian(20) breakout ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for stoploss and thresholds ===
    tr_4h = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                  np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for Donchian(20) and ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Detection from 12h ADX ---
        adx_val = adx_12h_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # --- Donchian Breakout Signals ---
        breakout_up = price > donchian_high[i-1]  # Break above previous period high
        breakout_down = price < donchian_low[i-1]  # Break below previous period low
        
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
                # Exit on opposite breakout in ranging market
                if is_ranging and breakout_down and volume_spike:
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
                # Exit on opposite breakout in ranging market
                if is_ranging and breakout_up and volume_spike:
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
        if volume_spike:
            if is_trending:
                # Trend mode: breakout continuation
                if breakout_up:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif breakout_down:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif is_ranging:
                # Range mode: mean reversion at Donchian edges
                if breakout_up:
                    # Price broke above Donchian high -> expect reversion -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif breakout_down:
                    # Price broke below Donchian low -> expect reversion -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # In transition regime (ADX 20-25), stay flat
        else:
            signals[i] = 0.0
    
    return signals