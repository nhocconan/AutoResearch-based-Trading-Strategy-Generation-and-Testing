#!/usr/bin/env python3
"""
Experiment #339: 6h Elder Ray + 1d Regime + Volume Spike

HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures 
buying/selling pressure relative to trend. Combined with 1d ADX regime filter and 
volume confirmation, this captures strong impulsive moves in both bull and bear markets. 
In trending markets (ADX > 25): enter on Bull/Bear power expansion with volume. 
In ranging markets (ADX < 25): fade extreme Elder Ray readings at Bollinger Bands. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_339_6h_elder_ray_1d_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime detection and Bollinger Bands ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 1d ADX for regime detection
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Calculate 1d Bollinger Bands (20, 2) for fading extremes in ranging markets
    sma20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_6h = high - ema13_1d_aligned
    bear_power_6h = low - ema13_1d_aligned
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 100  # Warmup for 1d indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter: ADX > 25 = trending, ADX < 25 = ranging ---
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price and Power Levels ---
        price = close[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        upper_bb = upper_bb_1d_aligned[i]
        lower_bb = lower_bb_1d_aligned[i]
        
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
                # Exit on mean reversion to EMA13 in ranging markets
                if is_ranging and abs(price - ema13_1d_aligned[i]) < 0.5 * atr_14[i]:
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
                # Exit on mean reversion to EMA13 in ranging markets
                if is_ranging and abs(price - ema13_1d_aligned[i]) < 0.5 * atr_14[i]:
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
        # Trending market logic: Elder Ray expansion with volume
        if is_trending:
            # Long: Bull Power expansion (making new highs) + volume spike
            long_entry = (bull_power > 0) and (bull_power == np.maximum.accumulate(bull_power_6h[:i+1])[-1]) and volume_spike
            # Short: Bear Power expansion (making new lows) + volume spike
            short_entry = (bear_power < 0) and (bear_power == np.minimum.accumulate(bear_power_6h[:i+1])[-1]) and volume_spike
        # Ranging market logic: Fade extreme Elder Ray at Bollinger Bands
        else:
            # Long: Bear Power extreme at lower BB (oversold) + volume spike
            long_entry = (bear_power < 0) and (price < lower_bb) and volume_spike
            # Short: Bull Power extreme at upper BB (overbought) + volume spike
            short_entry = (bull_power > 0) and (price > upper_bb) and volume_spike
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals