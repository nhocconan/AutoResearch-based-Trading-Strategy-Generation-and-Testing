#!/usr/bin/env python3
"""
Experiment #051: 6h Williams %R + 1d ADX Trend Filter + Volume Confirmation

HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
Trades are taken only when aligned with 1d ADX trend (ADX>25) and confirmed by volume spikes (>2.0x average). 
In trending markets (ADX>25), we fade extreme %R readings expecting continuation. 
In ranging markets (ADX<=25), we avoid trading to reduce false signals. 
The 6h timeframe targets 12-37 trades/year (50-150 total over 4 years). 
Volume confirmation ensures conviction. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_051_6h_williamsr_1d_adx_v1"
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
    
    # Calculate ADX(14) on 1d data
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = 0  # First period has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Movement
        up_move = np.diff(high_1d, prepend=high_1d[0])
        down_move = -np.diff(low_1d, prepend=low_1d[0])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
        minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
        adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF ADX and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade when 1d ADX > 25 (trending market) ---
        trending_market = adx_1d_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Williams %R Conditions ---
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
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
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold + volume spike + trending market
        long_condition = oversold and volume_spike and trending_market
        
        # Short: Williams %R overbought + volume spike + trending market
        short_condition = overbought and volume_spike and trending_market
        
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