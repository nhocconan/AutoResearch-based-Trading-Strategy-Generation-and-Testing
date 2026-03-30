#!/usr/bin/env python3
"""
Experiment #009: 4h ATR Channel Breakout + 1d KAMA Trend + Volume

HYPOTHESIS: ATR Channels adapt to volatility and identify true breakouts vs
range noise. Combined with 1d KAMA trend filter and volume confirmation,
this catches high-probability trend continuations.

WHY 4h: 41% keep rate in DB. Balances trade frequency and signal quality.
ATR channels on 4h capture multi-day swings with proper volatility adjustment.

WHY IT WORKS: ATR channels expand in volatile markets, contract in calm.
KAMA adapts to trend strength. Volume confirms institutional interest.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atrchannel_kama_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using proper EWM"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10):
    """KAMA - Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Efficiency Ratio = change / volatility
    change = np.abs(close[period:] - close[:-period])
    
    # Volatility = sum of |price changes| over period
    volatility = np.zeros(n - period)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+period+1])))
    
    er = np.zeros(n)
    er[period:] = change / (volatility + 1e-10)
    
    # SC = (ER * (fast - slow) + slow)^2
    fast, slow = 2 / (2 + 1), 2 / (30 + 1)
    sc = (er * (fast - slow) + slow) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend (period=10 adapts faster)
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR Channel (multiplier 2.0 = moderate band width)
    upper_atr = close + 2.0 * atr_14
    lower_atr = close - 2.0 * atr_14
    
    # Previous bar's channel for breakout detection
    prev_upper = np.roll(upper_atr, 1)
    prev_lower = np.roll(lower_atr, 1)
    prev_upper[0] = upper_atr[0]
    prev_lower[0] = lower_atr[0]
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(100, period * 2 + 1)  # Buffer for KAMA and ATR
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA) ===
        price_above_kama = close[i] > kama_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # ATR Channel breakout detection (previous bar's channel)
        upper_break = high[i] > prev_upper[i]  # Price broke above upper ATR channel
        lower_break = low[i] < prev_lower[i]   # Price broke below lower ATR channel
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Break above upper ATR channel + uptrend + volume ===
            if price_above_kama and upper_break and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Break below lower ATR channel + downtrend + volume ===
            if not price_above_kama and lower_break and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 R:R) ===
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= 4:  # Hold at least 4 bars (16h) before TP check
                if position_side > 0:
                    profit_target = entry_price + 3.0 * atr_14[i]
                    if close[i] >= profit_target:
                        desired_signal = 0.0
                if position_side < 0:
                    profit_target = entry_price - 3.0 * atr_14[i]
                    if close[i] <= profit_target:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals