#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Use 12h primary with 1w HTF for structural trend filtering.
In bear markets (2022 crash), 1w HMA stays below price longer, reducing longs.
In bull markets (2021, 2025+), 1w HMA above price supports longs.
This should generate ~75-150 trades total (18-37/year), within proven 12h range.

KEY INSIGHT: DB winners all use HTF trend confirmation. Using 1w vs 12h gives:
- Slower trend confirmation = fewer but higher quality signals
- 1w is "macro enough" to filter 2022 bear (-77%) while allowing 2021/2025 rallies

RULES:
1. 12h Donchian(20) breakout (structural price channel, proven pattern)
2. 1w HMA(21) > close for longs, < close for shorts (macro trend filter)
3. Volume spike 2.0x above 20-ema (confirmation, not noise)
4. 12h ATR 2.5x stoploss (dynamic, adapts to volatility)
5. Min hold 8 bars (prevents fee churn on 12h TF)

TARGET: 75-150 trades total over 4 years (~20-40/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_hma_vol_v1"
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

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_momentum(close, period=10):
    """Simple momentum: rate of change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    return (close / close[period:].mean()) * 100 - 100  # simplified

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA(21) for macro trend (very slow, structural)
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for bear protection)
    
    # Position tracking
    position_side = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 300  # 200 donchian + 20 vol MA + 80 for HTF alignment buffer
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_aligned[i]  # price above 1w HMA = bull
        macro_bear = close[i] < hma_aligned[i]  # price below 1w HMA = bear
        
        # === DONCHIAN BREAKOUT (previous bar close vs channel) ===
        prev_close = close[i - 1]
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = prev_close > prev_donchian_up  # yesterday closed above channel high
        breakout_down = prev_close < prev_donchian_lo  # yesterday closed below channel low
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] >= 2.0  # 2x average volume
        
        # === MOMENTUM CONFIRMATION (10-bar rate of change) ===
        if i >= 10:
            momentum = (close[i] / close[i - 10] - 1) * 100
            strong_momentum = abs(momentum) > 1.5  # 1.5% in 10 bars
        else:
            strong_momentum = False
        
        # === MINIMUM HOLD: 8 bars for 12h TF (at least 4 days) ===
        bars_held = i - entry_bar
        min_hold_ok = bars_held >= 8
        
        # === EXIT CONDITIONS ===
        if position_side != 0:
            # Long exit conditions
            if position_side == 1:
                # Stoploss: price fell 2.5 ATR from entry
                if low[i] < entry_price - 2.5 * entry_atr:
                    position_side = 0
                    trailing_high = 0.0
                
                # Macro trend flip to bear
                elif macro_bear:
                    position_side = 0
                    trailing_high = 0.0
                
                # Donchian breakdown signal
                elif prev_close < prev_donchian_lo:
                    position_side = 0
                    trailing_high = 0.0
            
            # Short exit conditions
            elif position_side == -1:
                # Stoploss: price rose 2.5 ATR from entry
                if high[i] > entry_price + 2.5 * entry_atr:
                    position_side = 0
                    trailing_low = 0.0
                
                # Macro trend flip to bull
                elif macro_bull:
                    position_side = 0
                    trailing_low = 0.0
                
                # Donchian breakout signal
                elif prev_close > prev_donchian_up:
                    position_side = 0
                    trailing_low = 0.0
        
        # === ENTRY CONDITIONS ===
        if position_side == 0:
            signal_value = 0.0
            
            # LONG: Macro bull + breakout up + volume spike + momentum
            if macro_bull and breakout_up and vol_spike:
                signal_value = SIZE
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
            
            # SHORT: Macro bear + breakout down + volume spike + momentum
            elif macro_bear and breakout_down and vol_spike:
                signal_value = -SIZE
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_low = low[i]
            
            # Update position
            position_side = int(np.sign(signal_value))
            signals[i] = signal_value
        
        else:
            # In position - hold until exit condition met
            # Keep signal active if position held
            signals[i] = position_side * SIZE
    
    return signals