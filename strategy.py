#!/usr/bin/env python3
"""
Experiment #360: 6h Elder Ray + ADX Regime + Volume Spike

HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure, 
combined with ADX regime filter (trending vs ranging) and 12h volume spike confirmation. 
In trending markets (ADX > 25), we take Elder Ray signals in direction of trend. 
In ranging markets (ADX < 20), we fade extreme Elder Ray readings. 
Volume spike confirms institutional participation. Targets 12-37 trades/year on 6h timeframe 
by requiring confluence of three filters, reducing overtrading while capturing high-probability 
setups in both bull and bear markets through regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        up_move = high_1d - np.roll(high_1d, 1)
        down_move = np.roll(low_1d, 1) - low_1d
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging
    
    # === 6h Indicators ===
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Signals Initialization
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: ADX > 25 = trending, ADX < 20 = ranging ---
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trending Market Logic: Follow Elder Ray in direction of trend
        if is_trending and volume_spike:
            # Determine 1d trend direction using EMA50 slope
            if len(df_1d) >= 50:
                close_1d = df_1d['close'].values
                ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
                ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
                
                # Uptrend if price > EMA50 and rising
                if i > 0 and not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
                    uptrend = close[i] > ema_50_1d_aligned[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
                    downtrend = close[i] < ema_50_1d_aligned[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
                else:
                    uptrend = close[i] > ema_50_1d_aligned[i] if not np.isnan(ema_50_1d_aligned[i]) else False
                    downtrend = close[i] < ema_50_1d_aligned[i] if not np.isnan(ema_50_1d_aligned[i]) else False
            else:
                uptrend = close[i] > np.mean(close[max(0, i-50):i]) if i >= 50 else False
                downtrend = close[i] < np.mean(close[max(0, i-50):i]) if i >= 50 else False
            
            # Long: Strong Bull Power in uptrend
            long_condition = uptrend and (bull_power[i] > np.percentile(bull_power[max(0, i-100):i], 80) if i >= 100 else bull_power[i] > 0)
            # Short: Strong Bear Power in downtrend
            short_condition = downtrend and (bear_power[i] < np.percentile(bear_power[max(0, i-100):i], 20) if i >= 100 else bear_power[i] < 0)
            
        # Ranging Market Logic: Fade extreme Elder Ray readings
        elif is_ranging and volume_spike:
            # Long: Extreme Bear Power (oversold) 
            long_condition = bear_power[i] < np.percentile(bear_power[max(0, i-100):i], 15) if i >= 100 else bear_power[i] < np.mean(bear_power) - np.std(bear_power)
            # Short: Extreme Bull Power (overbought)
            short_condition = bull_power[i] > np.percentile(bull_power[max(0, i-100):i], 85) if i >= 100 else bull_power[i] > np.mean(bull_power) + np.std(bull_power)
        else:
            long_condition = False
            short_condition = False
        
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