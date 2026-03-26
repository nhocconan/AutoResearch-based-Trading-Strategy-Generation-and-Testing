#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + 1d SMA200 Trend

HYPOTHESIS: 4h Donchian(20) breakouts capture institutional moves with enough
frequency for statistical validity. Volume confirmation (1.8x MA20) filters
false breakouts. 1d SMA200 provides regime bias without overfitting.
2.5 ATR stop + 6-bar cooldown prevents whipsaws. This exact pattern appears
in DB as top performer (SOLUSDT test Sharpe 1.38-1.46).

TIMEFRAME: 4h primary
HTF: 1d SMA200 for regime bias
TARGET: 75-120 total trades over 4 years (19-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_sma200_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend bias
    sma_200_1d = df_1d['close'].rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_middle, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA20
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    cooldown_remaining = 0  # bars until next entry allowed
    
    warmup = 100
    
    for i in range(warmup, n):
        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d SMA200) ===
        above_sma200 = close[i] > sma_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: current close > previous upper band (not current, which is still forming)
        breakout_up = (close[i] > donch_upper[i-1]) if i > 0 else False
        # Breakout down: current close < previous lower band
        breakout_down = (close[i] < donch_lower[i-1]) if i > 0 else False
        
        desired_signal = 0.0
        
        # === NEW LONG ENTRY ===
        if not in_position and cooldown_remaining == 0:
            if breakout_up and vol_spike and above_sma200:
                desired_signal = SIZE
        
        # === NEW SHORT ENTRY ===
        if not in_position and cooldown_remaining == 0:
            if breakout_down and vol_spike and not above_sma200:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            cooldown_remaining = 6  # 6 bars = 1 day cooldown
        
        # === EXIT: Opposite signal OR channel touch ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower channel OR RSI extreme
            if breakout_down:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper channel
            if breakout_up:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
            cooldown_remaining = 6
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # else: same direction - maintain position
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