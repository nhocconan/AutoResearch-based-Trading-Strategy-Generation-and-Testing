#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian(20) Breakout + 1d Trend + Volume Spike + Choppiness Filter

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, when aligned with 1d trend (price above/below 1d EMA50),
confirmed by volume spikes (>1.5x 20-bar MA), and filtered by choppiness regime (CHOP > 38.2 = trending),
capture strong trending moves in both bull and bear markets. Uses ATR-based trailing stop (2.5x ATR).
Designed for low trade frequency (~19-50/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_trend_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: values > 61.8 = ranging, < 38.2 = trending."""
    n = len(close)
    if n < period:
        return np.full(n, 50.0)
    
    atr_sum = np.zeros(n)
    tr = np.zeros(n)
    for i in range(n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period-1, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.full(n, 50.0)
    for i in range(period-1, n):
        if atr_sum[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === HTF: 1d Choppiness for regime filter (Call ONCE before loop) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        ema_50_1d = ema_50_1d_aligned[i]
        chop_1d_val = chop_1d_aligned[i]
        
        # --- 1d Trend Filter ---
        trend_bullish = close[i] > ema_50_1d
        trend_bearish = close[i] < ema_50_1d
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 38.2) ---
        trending_market = chop_1d_val < 38.2
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal OR choppy regime
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~8h)
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR market becomes choppy
                    if trend_bearish or not trending_market:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR market becomes choppy
                    if trend_bullish or not trending_market:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: Breakout above DC upper with bullish 1d trend AND volume confirmation AND trending market
        if bullish_breakout and trend_bullish and vol_ok and trending_market:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions: Breakout below DC lower with bearish 1d trend AND volume confirmation AND trending market
        elif bearish_breakout and trend_bearish and vol_ok and trending_market:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals