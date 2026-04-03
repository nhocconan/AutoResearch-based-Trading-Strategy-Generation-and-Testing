#!/usr/bin/env python3
"""
Experiment #074: 1h Donchian(20) Breakout + 4h Trend + Volume Spike + Session Filter

HYPOTHESIS: 1h Donchian breakouts aligned with 4h price trend (close > SMA50) and 
volume confirmation (1.5x average volume) capture momentum moves. Session filter 
(08-20 UTC) reduces noise. Using 4h for direction and 1h for timing targets 15-37 
trades/year. Discrete position sizing (0.20) minimizes fee drag. ATR-based stop 
manages risk.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_1h_donchian_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h SMA50 for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    sma_4h_50 = pd.Series(df_4h['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, sma_4h_50)
    
    # === 1h Indicators ===
    atr_14 = pd.Series(high - low).ewm(span=14, min_periods=14, adjust=False).mean().values
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(sma_4h_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 4h Trend Filter ---
        trend_bullish = close[i] > sma_4h_50_aligned[i]
        trend_bearish = close[i] < sma_4h_50_aligned[i]
        
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
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~2h)
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
        # Breakout above upper Donchian with bullish 4h trend, volume confirmation
        if bullish_breakout and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 4h trend, volume confirmation
        elif bearish_breakout and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals