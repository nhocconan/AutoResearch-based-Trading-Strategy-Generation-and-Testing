#!/usr/bin/env python3
"""
Experiment: 4h ATR Volatility Breakout + Donchian + 1d HMA

HYPOTHESIS: Markets move in cycles of contraction (low ATR) → expansion (high ATR).
By detecting the START of volatility expansion (ATR ratio > 1.5) with price breaking
out of 4h Donchian channel AND 1d HMA trend alignment, we catch moves at their infancy.

WHY IT WORKS: ATR ratio breakout is a momentum/volatility combo signal.
Unlike price-only breakouts, this confirms institutional participation (vol spike).
1d HMA(21) is slow enough to avoid whipsaws, fast enough to catch 2-4 week moves.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atr_donchian_vol_hma21_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with EWM smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = pd.Series(data).rolling(window=period // 2, min_periods=period // 2).mean()
    full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    hma = (2 * half - full)
    hma = hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio: current ATR vs 30-bar ATR MA (detects vol expansion)
    atr_ma30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where((atr_ma30 > 0) & ~np.isnan(atr_ma30), atr_ma30, 1)
    
    # 4h Donchian channel (20 periods = 5 days)
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume ratio
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
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
    
    warmup = 100  # Buffer for alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === ENTRY CONDITIONS ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # Volatility expansion (>1.5x normal)
        vol_expansion = atr_ratio[i] > 1.5
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout (close above/below channel)
        donch_breakout_up = close[i] > donchian_upper[i]
        donch_breakout_down = close[i] < donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Volatility expansion + Donchian breakout + HMA trend ===
            if price_above_hma and vol_expansion and vol_spike:
                if donch_breakout_up:
                    desired_signal = SIZE
            
            # === SHORT: Volatility expansion + Donchian breakout + HMA trend ===
            if price_below_hma and vol_expansion and vol_spike:
                if donch_breakout_down:
                    desired_signal = -SIZE
        
        # === TRAILING STOP (2 ATR) ===
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
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 3:
            # Only allow exit via stoploss, not take profit
            if desired_signal == 0.0 and position_side > 0 and low[i] >= stop_price:
                desired_signal = position_side * SIZE
            if desired_signal == 0.0 and position_side < 0 and high[i] <= stop_price:
                desired_signal = position_side * SIZE
        
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