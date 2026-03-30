#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + Volume + 1w Trend

HYPOTHESIS: 1d timeframe with 1w macro filter is designed for swing trading
- 1d = ~365 bars/year, Donchian(20) breakout ≈ every 20-40 days = 9-18 signals/year
- 1w SMA(50) as macro trend filter prevents fighting major trends
- Volume confirmation prevents false breakouts
- 2.5x ATR trailing stop for risk management
- Min 3-bar hold prevents whipsaw

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull (2021, late 2023-2024): Long breakouts above 1w SMA
- Bear (2022): Short breakdowns below 1w SMA, fades rallies
- Range (2025): Fewer signals = adaptive, smaller drawdowns

TARGET: 30-100 total trades over 4 years (7-25/year). Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)  # auto shift(1)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 - price channel breakout
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_since_exit = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === BREAKOUT CONDITIONS ===
        bullish_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma[i] * 1.3 if vol_ma[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            signals[i] = position_side * SIZE
        
        # === EXIT CONDITIONS ===
        if in_position:
            stop_hit = False
            
            # ATR trailing stop (2.5x)
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Min hold: 3 bars (prevents whipsaw)
            min_hold = (i - entry_bar) >= 3
            
            # HTF trend reversal exit (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                bars_since_exit = 0
        
        # === NEW POSITIONS ===
        if not in_position:
            bars_since_exit += 1
            # Cooldown: wait 2 bars after exit before new entry
            if bars_since_exit < 2:
                signals[i] = 0.0
                continue
            
            # LONG: Breakout above + volume confirm + 1w uptrend
            if bullish_breakout and vol_ok and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakdown below + volume confirm + 1w downtrend
            elif bearish_breakout and vol_ok and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals