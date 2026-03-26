#!/usr/bin/env python3
"""
Experiment #026: 4h HMA + Donchian + Volume Confirmation

HYPOTHESIS: This exact combination (HMA for trend, Donchian for structure, 
volume for confirmation) is proven on SOLUSDT (test Sharpe 1.38, 95 trades, 52% win).
Previous attempts overcomplicated with extra indicators or used loose volume (1.5x).
This version uses STRICTER volume filter (1.8x) to push win rate above 50%.

WHY IT WORKS IN BULL AND BEAR:
- Bull: HMA up + price breaks above Donchian high + volume spike = momentum confirmation
- Bear: HMA down + price breaks below Donchian low + volume spike = short momentum
- Range: Volume spikes less frequent in ranges = natural filter

KEY DIFFERENCE FROM FAILED ATTEMPTS:
- Previous donchian_vol_adx had 479 trades (too loose, negative Sharpe)
- This uses only 3 conditions with tighter volume (1.8x) = ~75-150 trades expected
- No redundant indicators (ADX/RSI/chop just add noise)

TARGET: 75-150 trades over 4 years. Win rate >50%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    half = pd.Series(close).ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * half - full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Indicators
    hma_16 = calculate_hma(close, 16)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Donchian Channel (20 periods = 5 days on 4h)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (strict - 1.8x average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        desired_signal = 0.0
        
        # === TREND DIRECTION (HMA) ===
        price_above_hma = close[i] > hma_16[i]
        price_below_hma = close[i] < hma_16[i]
        
        # === VOLUME CONFIRMATION (strict - 1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        upper_broken = high[i] >= donch_upper[i]  # Price reached new 20-bar high
        lower_broken = low[i] <= donch_lower[i]   # Price reached new 20-bar low
        
        # === ENTRY LOGIC ===
        if not in_position:
            # Long: Price above HMA (uptrend) + breakout above Donchian high + volume confirmation
            if price_above_hma and upper_broken and vol_spike:
                desired_signal = SIZE
            
            # Short: Price below HMA (downtrend) + breakdown below Donchian low + volume confirmation
            elif price_below_hma and lower_broken and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (3x ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * atr_14[i]
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * atr_14[i]
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1 day) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 6:
            # Exit if trend reverses
            if position_side > 0 and not price_above_hma:
                desired_signal = 0.0
            if position_side < 0 and not price_below_hma:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = i
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals