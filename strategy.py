#!/usr/bin/env python3
"""
Experiment #062: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: 12h Donchian breakouts aligned with 1d Hull Moving Average trend and 
volume confirmation (1.5x average volume) capture strong momentum moves with 
appropriate trade frequency for 12h timeframe (target: 12-37 trades/year). 
Uses ATR-based trailing stoploss (2.0x) for risk management. Designed to work 
in both bull and bear markets by using trend-following breakouts with volume 
confirmation to avoid false signals. Discrete position sizing (0.25) minimizes 
fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_volume_v1"
timeframe = "12h"
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

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # === 12h Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_1d_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d HMA Trend Filter ---
        trend_bullish = close[i] > hma_1d_21_aligned[i]
        trend_bearish = close[i] < hma_1d_21_aligned[i]
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1 day)
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR price touches lower Donchian
                    if trend_bearish or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR price touches upper Donchian
                    if trend_bullish or close[i] >= dc_upper_20[i]:
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
        # Breakout above upper Donchian with bullish 1d HMA trend AND volume confirmation
        if bullish_breakout and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 1d HMA trend AND volume confirmation
        elif bearish_breakout and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals