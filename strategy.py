#!/usr/bin/env python3
"""
Experiment #004: 1d KAMA + RSI + 1w Choppiness Regime

HYPOTHESIS: KAMA adapts to volatility conditions better than fixed MAs.
Combined with RSI for momentum confirmation and 1w Choppiness to stay out
of range-bound markets, this captures medium-term directional moves.

WHY 1d: Very few trades (7-25/year target), minimal fee drag.
WHY 1w HTF: Weekly choppiness confirms structural trend vs. chop.
Keep rate target: 40%.

TARGET: 30-100 total trades over 4 years. HARD MAX: 150.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_004_1d_kama_rsi_1w_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(period):
            volatility += abs(close[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else 0
        er[i] = direction / volatility if volatility > 0 else 0
    
    # Fast and slow EMA constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (31 + 1)
    
    for i in range(1, n):
        if np.isnan(er[i]) or er[i] <= 0:
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.zeros(n)
    deltas[1:] = close[1:] - close[:-1]
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(period):
            idx = i - j
            if idx > 0:
                tr = max(high[idx] - low[idx], abs(high[idx] - close[idx - 1]))
            else:
                tr = high[idx] - low[idx]
            tr_sum += tr
        
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_hl = hh - ll
        
        if range_hl > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w Choppiness Index for regime filtering
    chop_1w = calculate_choppiness(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=13)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Local indicators
    kama = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    bars_since_signal = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === KAMA TREND ===
        kama_rising = kama[i] > kama[i - 1] if i > 0 and not np.isnan(kama[i - 1]) else False
        kama_falling = kama[i] < kama[i - 1] if i > 0 and not np.isnan(kama[i - 1]) else False
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === RSI MOMENTUM ===
        rsi_oversold = rsi[i] < 35  # Oversold bounce
        rsi_overbought = rsi[i] > 65  # Overbought rejection
        rsi_neutral_high = rsi[i] > 50  # Confirming strength
        rsi_neutral_low = rsi[i] < 50  # Confirming weakness
        
        # === 1w REGIME (Choppiness) ===
        is_trending_1w = chop_1w_aligned[i] < 55  # Lower = trending
        is_choppy_1w = chop_1w_aligned[i] > 61.8  # Higher = choppy
        
        # Skip new entries in choppy markets
        if not in_position and is_choppy_1w:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR STOPLOSS LEVEL ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * entry_atr
                if low[i] < stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * entry_atr
                if high[i] > stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: KAMA rising + RSI oversold + trending regime ===
            # Price bouncing from oversold with KAMA confirming uptrend
            if kama_rising and rsi_oversold and is_trending_1w:
                desired_signal = SIZE
            # Alternative: KAMA rising + RSI neutral-high + volume spike
            elif kama_rising and rsi_neutral_high and price_above_kama and vol_spike:
                desired_signal = SIZE / 2  # Half size - less confident
        
        # === HOLDING: Check for exit conditions ===
        if in_position:
            bars_held = i - entry_bar
            
            # Exit long: KAMA stops rising OR RSI overbought
            if position_side > 0:
                if not kama_rising and bars_held >= 5:
                    desired_signal = SIZE / 2  # Reduce to half
                if kama_falling or (rsi[i] > 75 and bars_held >= 3):
                    desired_signal = 0.0  # Exit
            
            # Exit short: KAMA stops falling OR RSI oversold
            if position_side < 0:
                if not kama_falling and bars_held >= 5:
                    desired_signal = -SIZE / 2
                if kama_rising or (rsi[i] < 25 and bars_held >= 3):
                    desired_signal = 0.0  # Exit
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals