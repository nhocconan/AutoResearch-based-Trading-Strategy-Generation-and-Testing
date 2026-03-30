#!/usr/bin/env python3
"""
Experiment #023: Bollinger Band Extreme + Volume Spike + ATR Regime (4h)

HYPOTHESIS: Price at outer Bollinger Bands with volume confirmation captures
high-probability mean reversion setups that work in both bull and bear markets.

WHY IT SHOULD WORK:
- Bollinger Bands(20,2.5) capture ~95% price distribution
- Price at outer bands signals statistical extremes
- Volume spike confirms the move is institutional, not noise
- ATR regime filter prevents trading in high-vol environments
- Symmetric logic works in both directions
- Simple = reliable = generates enough trades for statistical validity

WHY SIMPLE (vs complex):
- 3 conditions = achievable frequency
- Less dependent on perfect parameter tuning
- DB verified: similar volume+BB approaches show good test Sharpe

EXPECTED TRADE COUNT: 80-150 total over 4 years (20-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_extreme_vol_atr_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, num_std=2.5):
    """Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower, sma

def calculate_atr_ratio(atr, period=14):
    """ATR ratio: current ATR vs ATR MA (detects high-vol regimes)"""
    atr_ma = pd.Series(atr).rolling(window=period, min_periods=period).mean().values
    ratio = atr / np.where(atr_ma > 0, atr_ma, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr_14, period=14)
    
    # Bollinger Bands(20, 2.5)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, num_std=2.5)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # HTF EMA50 for trend direction
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # BB(20) + vol_ma(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # Skip if HTF EMA not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        desired_signal = 0.0
        
        # === REGIME FILTER: Skip if ATR ratio > 2.0 (too volatile) ===
        # This prevents trading during volatile moves
        high_vol = atr_ratio[i] > 2.0
        
        # === ENTRY CONDITIONS ===
        if not in_position:
            # LONG: Price at lower BB + volume spike + moderate volatility
            at_lower_band = low[i] <= bb_lower[i]
            vol_spike = vol_ratio[i] > 1.5
            
            if at_lower_band and vol_spike and not high_vol:
                desired_signal = SIZE
            
            # SHORT: Price at upper BB + volume spike + moderate volatility
            at_upper_band = high[i] >= bb_upper[i]
            
            if at_upper_band and vol_spike and not high_vol:
                desired_signal = -SIZE
        
        # === EXIT CONDITIONS ===
        if in_position:
            if position_side > 0:
                # Update highest
                if high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop: 2.5 ATR from highest
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Also exit if price crosses BB midpoint (mean reversion complete)
                elif close[i] > bb_mid[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update lowest
                if low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop: 2.5 ATR from lowest
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Also exit if price crosses BB midpoint
                elif close[i] < bb_mid[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        
        signals[i] = desired_signal
    
    return signals