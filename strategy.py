#!/usr/bin/env python3
"""
Experiment #119: 6h Williams %R + 12h Volume Spike + 1d ADX Trend Filter

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
combined with 12h volume spike confirmation and 1d ADX trend filter (>25) to ensure 
trending market conditions. This strategy targets mean reversion in strong trends 
(Williams %R >80 for short in downtrend, <20 for long in uptrend) with volume 
confirmation to filter false signals. Designed for 12-37 trades/year on 6h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing high-probability 
reversals within established trends. Works in both bull and bear markets by 
aligning with higher timeframe direction via ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_vol_trend_v1"
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
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
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
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        
        # Align to LTF
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_1d_aligned = np.full(n, 0.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero (when high == low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
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
        if (np.isnan(williams_r[i]) or np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        trending_market = adx_1d_aligned[i] > 25
        
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
                # Take profit at Williams %R reversal (exit extreme)
                if williams_r[i] > -20:  # Exit overbought
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
                # Take profit at Williams %R reversal (exit extreme)
                if williams_r[i] < -80:  # Exit oversold
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold (< -80) in uptrend (we infer trend from price vs EMA50 proxy)
        # Use price vs 20-period EMA on 6h as short-term trend filter
        if i >= 20:
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            price_above_ema = close[i] > ema_20
            price_below_ema = close[i] < ema_20
        else:
            price_above_ema = True  # Default to allow trading during warmup
            price_below_ema = True
        
        long_condition = (
            williams_r[i] < -80 and  # Oversold
            price_above_ema and      # In short-term uptrend
            trending_market and      # Higher timeframe trend confirmed
            volume_spike             # Volume confirmation
        )
        
        # Short: Williams %R overbought (> -20) in downtrend
        short_condition = (
            williams_r[i] > -20 and  # Overbought
            price_below_ema and      # In short-term downtrend
            trending_market and      # Higher timeframe trend confirmed
            volume_spike             # Volume confirmation
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