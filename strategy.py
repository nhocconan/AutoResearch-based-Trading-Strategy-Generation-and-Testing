#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian Breakout + Volume + 1d HMA + Cooldown

HYPOTHESIS: 12h Donchian(20) breakouts mark institutional levels. Volume spike
confirms the move. 1d HMA filters direction. KEY FIX from prior failures: 
cooldown after exit prevents re-entry on same breakout (caused overtrading).
Target 75-150 total trades over 4 years.

TIMEFRAME: 12h
HTF: 1d HMA for trend bias
ENTRY: Donchian breakout + volume spike + 1d HMA aligned
EXIT: Opposite channel touch OR RSI extreme (25/75)
STOP: 2.5 ATR trailing
COOLDOWN: 3 bars after exit before re-entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_hma_cooldown_v1"
timeframe = "12h"
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

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_middle, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
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
    
    # Cooldown tracking (KEY FIX to prevent overtrading)
    cooldown_bars = 0
    last_exit_bar = -999
    
    warmup = 50
    
    for i in range(warmup, n):
        # Cooldown counter
        if cooldown_bars > 0:
            cooldown_bars -= 1
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Get current values
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i] > 1.5
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # Donchian breakout detection (cross of previous channel)
        # Breakout up: close crosses above previous upper
        breakout_up = False
        breakout_down = False
        if i > 1 and not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            if close[i-1] <= donch_upper[i-1] and close[i] > donch_upper[i-1]:
                breakout_up = True
            if close[i-1] >= donch_lower[i-1] and close[i] < donch_lower[i-1]:
                breakout_down = True
        
        # Price outside channel (for exit)
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === POSITION MANAGEMENT ===
        if in_position:
            # Update highest/lowest
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
            
            # Stop loss check
            stop_triggered = False
            if position_side > 0 and low[i] < stop_price:
                stop_triggered = True
            if position_side < 0 and high[i] > stop_price:
                stop_triggered = True
            
            if stop_triggered:
                in_position = False
                position_side = 0
                last_exit_bar = i
                cooldown_bars = 3  # Wait 3 bars before re-entry
                signals[i] = 0.0
                continue
            
            # Exit checks
            exit_triggered = False
            
            if position_side > 0:
                # Long exit: price below lower channel OR RSI < 25
                if price_below_lower or rsi_val < 25:
                    exit_triggered = True
            else:
                # Short exit: price above upper channel OR RSI > 75
                if price_above_upper or rsi_val > 75:
                    exit_triggered = True
            
            if exit_triggered:
                in_position = False
                position_side = 0
                last_exit_bar = i
                cooldown_bars = 3  # Wait 3 bars before re-entry
                signals[i] = 0.0
                continue
            
            # Maintain position
            signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # === NEW ENTRY LOGIC ===
        if cooldown_bars > 0:
            signals[i] = 0.0
            continue
        
        # === LONG ENTRY ===
        if breakout_up and vol_spike and price_above_1d_hma:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            stop_price = entry_price - 2.5 * entry_atr
            signals[i] = SIZE
            continue
        
        # === SHORT ENTRY ===
        if breakout_down and vol_spike and not price_above_1d_hma:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            stop_price = entry_price + 2.5 * entry_atr
            signals[i] = -SIZE
            continue
        
        # No signal
        signals[i] = 0.0
    
    return signals