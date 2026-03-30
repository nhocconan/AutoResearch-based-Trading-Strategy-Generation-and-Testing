#!/usr/bin/env python3
"""
Experiment #007: 6h Fisher Transform + Donchian Breakout + 1d EMA Regime

HYPOTHESIS: Fisher Transform provides genuine mean-reversion signals that are
orthogonal to trend-following approaches (Donchian, ADX, HMA). Combined with
structural breakouts (Donchian) and regime filter (1d EMA), this creates a
strategy that catches reversals during 2022 crash rallies and 2025 bear rallies.

WHY GENUINELY DIFFERENT FROM 16,000 EXPERIMENTS:
- Fisher Transform transforms price into Gaussian-like distribution
- Entry when Fisher crosses signal thresholds, NOT trend-following crossovers
- RSI/ADX/HMA all measure directional momentum - Fisher measures REVERSAL probability
- Tested Fisher combinations work on 1d (program.md Tier 4), scaling to 6h

WHY IT SHOULD WORK IN BOTH MARKETS:
- Fisher extreme zones (|F| > 2.0) catch reversal points in trending moves
- 2022 crash had multiple sharp rallies (Fisher spikes) - catch these bounces
- 2025 bear has range-bound moves with occasional breaks - Fisher confirms exhaustion
- 1d EMA filter prevents fighting the larger trend

TRADE COUNT ESTIMATE:
- Fisher |F| > 1.5: ~20% of bars
- Donchian breakout confirm: ~40% pass
- 1d EMA aligned: ~60% pass
- Volume spike: ~70% pass
- Expected: ~3-4% of bars trigger = ~50-70 trades/year at 6h
- 4yr total: ~200-280 trades - in target range (75-250)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_donchian_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=10):
    """
    Fisher Transform - converts price to Gaussian-like distribution
    Formula: FISH = 0.5 * ln((1+G)/(1-G))
    Where G = 2 * ((high-low)/(high-low)) - 1 normalized over period
    
    Signals:
    - FISH crosses above trigger line -> bullish
    - FISH crosses below trigger line -> bearish
    - |FISH| > 2.0 -> extreme reversal zone
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate normalized price position
    hl2 = (high + low) / 2.0
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_arr = highest - lowest
    range_arr = np.where(np.abs(range_arr) < 1e-10, 1.0, range_arr)
    
    # Normalize: value between -1 and +1
    value = np.zeros(n)
    for i in range(period - 1, n):
        if not np.isnan(highest[i]) and not np.isnan(lowest[i]):
            val = (hl2[i] - lowest[i]) / (highest[i] - lowest[i])
            val = 2.0 * val - 1.0
            val = np.clip(val, -0.9999, 0.9999)
            value[i] = val
    
    # Fisher Transform
    fish = np.zeros(n)
    prev_fish = 0.0
    for i in range(n):
        if abs(value[i]) < 0.9999:
            fish[i] = 0.5 * np.log((1.0 + value[i]) / (1.0 - value[i]))
            prev_fish = fish[i]
        else:
            # Extrapolate if value hits limits
            fish[i] = prev_fish
    
    # Trigger line = lagged Fisher
    trigger = pd.Series(fish).shift(1).fillna(0).values
    
    return fish, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for trend regime (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, trigger = calculate_fisher_transform(high, low, close, period=10)
    
    # Donchian Channel(20) for structural breakout
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Sufficient for Fisher(10), Donchian(20), EMA21 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1d EMA REGIME FILTER ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === FISHER CROSS SIGNALS ===
        # Bullish: Fisher crosses above trigger (prev was below)
        fisher_bull_cross = (fisher[i] > trigger[i]) and (fisher[i-1] <= trigger[i-1]) if i > 0 else False
        
        # Bearish: Fisher crosses below trigger (prev was above)
        fisher_bear_cross = (fisher[i] < trigger[i]) and (fisher[i-1] >= trigger[i-1]) if i > 0 else False
        
        # Fisher in extreme zone (reversal probability high)
        fisher_extreme_bull = fisher[i] < -1.5  # Oversold - potential reversal up
        fisher_extreme_bear = fisher[i] > 1.5   # Overbought - potential reversal down
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bull_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        bear_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 4 bars (24h minimum) ===
        min_hold = (i - entry_bar) >= 4
        
        # === ATR FILTER: price moved enough ===
        atr_move = abs(close[i] - close[i-1]) if i > 0 else 0.0
        significant_move = atr_move > 0.5 * atr_14[i]
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Exit on opposite signal with min hold
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold:
                # Exit on Fisher reversal
                if position_side > 0 and fisher_bear_cross:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                elif position_side < 0 and fisher_bull_cross:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = position_side * SIZE
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Need Fisher signal + Donchian confirm + HTF aligned + volume
            
            # LONG: Fisher bullish cross + extreme oversold + HTF bullish + breakout + volume
            long_conditions = (
                (fisher_bull_cross and fisher_extreme_bull) and  # Fisher reversal
                htf_bullish and  # HTF aligned
                (bull_breakout or significant_move) and  # Price confirmation
                vol_spike  # Volume confirmation
            )
            
            # SHORT: Fisher bearish cross + extreme overbought + HTF bearish + breakdown + volume
            short_conditions = (
                (fisher_bear_cross and fisher_extreme_bear) and  # Fisher reversal
                htf_bearish and  # HTF aligned
                (bear_breakout or significant_move) and  # Price confirmation
                vol_spike  # Volume confirmation
            )
            
            if long_conditions:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            elif short_conditions:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals