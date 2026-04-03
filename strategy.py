#!/usr/bin/env python3
"""
Experiment #053: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation + Chop Filter

HYPOTHESIS: 4h Donchian breakouts aligned with 12h Hull Moving Average trend, 
volume confirmation (1.5x average volume), and choppiness regime filter (CHOP > 38.2 = trending) 
capture strong momentum moves while avoiding range-bound whipsaws. 
Uses ATR-based trailing stoploss (2.0x) for risk management. 
Designed for 25-40 trades/year to minimize fee drag while maintaining statistical significance. 
Discrete position sizing (0.25) reduces churn from minor signal fluctuations.
Works in both bull and bear markets by using trend-following logic with proper filters.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(values, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_full = pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean()
    wma_half = pd.Series(values).ewm(span=half, min_periods=half, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

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

def calculate_chop(high, low, close, period=14):
    """Choppiness Index: measures whether market is choppy (ranging) or not (trending).
    Values > 61.8 = ranging, < 38.2 = trending."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr_sum = np.zeros(n)
    for i in range(period-1, n):
        atr_sum[i] = np.sum([
            max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) 
            for j in range(i - period + 1, i + 1)
        ])
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(period-1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    chop = np.zeros(n)
    for i in range(period-1, n):
        if atr_sum[i] > 0 and highest_high[i] > lowest_low[i]:
            log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i]))
            chop[i] = 100 * log_val / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h HMA for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # === HTF: 12h Chop for regime filter (Call ONCE before loop) ===
    chop_12h = calculate_chop(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
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
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_12h_21_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h HMA Trend Filter ---
        trend_bullish = close[i] > hma_12h_21_aligned[i]
        trend_bearish = close[i] < hma_12h_21_aligned[i]
        
        # --- 12h Chop Regime Filter (trending market preferred) ---
        chop_ok = chop_12h_aligned[i] < 38.2  # Trending regime (below 38.2)
        
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
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal, chop regime change, or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~8h)
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR chop becomes ranging OR price touches lower Donchian
                    if trend_bearish or not chop_ok or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR chop becomes ranging OR price touches upper Donchian
                    if trend_bullish or not chop_ok or close[i] >= dc_upper_20[i]:
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
        # Long conditions: 
        # Breakout above upper Donchian with bullish 12h HMA trend AND trending chop AND volume confirmation
        if bullish_breakout and trend_bullish and chop_ok and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 12h HMA trend AND trending chop AND volume confirmation
        elif bearish_breakout and trend_bearish and chop_ok and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals