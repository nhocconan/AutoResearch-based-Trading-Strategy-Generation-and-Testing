#!/usr/bin/env python3
"""
Experiment #006: 1d KAMA Trend + ATR Volatility Regime + Donchian Breakout

HYPOTHESIS: Combining KAMA (adaptive trend) with ATR regime filtering and 
Donchian breakout captures sustained trends while avoiding whipsaws in 
high volatility periods. Works in BOTH bull and bear markets because 
KAMA adapts to direction automatically.

WHY 1d: Slowest timeframe = fewest trades = lowest fee drag = best generalization.
- Target: 50-150 total trades over 4 years (12-37/year)
- At 1d, even 50 trades can capture major market moves

WHY IT WORKS: 
- KAMA filters noise with adaptive smoothing
- ATR regime avoids entering during extreme volatility (volatility clustering)
- Donchian breakout confirms momentum with clear structure
- Simple 3 conditions = disciplined entries

PATTERN: KAMA bull → Donchian high breakout + vol regime → LONG
         KAMA bear → Donchian low breakout + vol regime → SHORT
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_atrregime_v1"
timeframe = "1d"
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

def calculate_kama(prices, period=21, fast_ema=2, slow_ema=30):
    """Kaufman Adaptive Moving Average"""
    n = len(prices)
    if n < slow_ema + 1:
        return np.full(n, np.nan)
    
    # Price changes
    price_change = np.abs(np.diff(prices, prepend=prices[0]))
    
    # Volatility (sum of price changes over period)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(price_change[i-period+1:i+1])
    
    # Efficiency ratio (ER)
    er = np.zeros(n)
    mask = volatility > 1e-10
    er[mask] = price_change[mask] / volatility[mask]
    
    # Smoothing constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = prices[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower arrays"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    # No HTF needed for this strategy (using 1d as primary)
    # All calculations on 1d data
    
    # === LOCAL 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR volatility regime: ratio of short to long ATR
    # High ratio (>1.5) = high volatility = avoid entries
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # KAMA for trend
    kama_21 = calculate_kama(close, period=21, fast_ema=2, slow_ema=30)
    
    # Donchian channels
    donch_upper_20, donch_mid_20, donch_lower_20 = calculate_donchian(high, low, period=20)
    donch_upper_55, donch_mid_55, donch_lower_55 = calculate_donchian(high, low, period=55)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for KAMA, Donchian alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(donch_upper_20[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK: Skip high volatility periods ===
        # ATR ratio > 1.5 means volatility expanding = higher whipsaw risk
        in_high_vol = atr_ratio[i] > 1.5
        
        # === TREND DIRECTION (KAMA) ===
        kama_bull = close[i] > kama_21[i]
        kama_bear = close[i] < kama_21[i]
        
        # === CONFIRMATION: Price above/below Donchian mid ===
        above_mid = close[i] > donch_mid_20[i]
        below_mid = close[i] < donch_mid_20[i]
        
        # === BREAKOUT: Close above 20-bar high (for longs) or below 20-bar low (for shorts) ===
        bull_breakout = close[i] > donch_upper_20[i]
        bear_breakout = close[i] < donch_lower_20[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === STOPLOSS: 2.5 ATR from entry ===
        # Trailing stop: maintain 2.5*ATR from highest/lowest since entry
        highest_since_entry = high[i]
        lowest_since_entry = low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Conditions: KAMA bull + close above Donchian mid + breakout + vol confirm + NOT high vol
            if kama_bull and above_mid and bull_breakout and vol_confirm and not in_high_vol:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Conditions: KAMA bear + close below Donchian mid + breakout + vol confirm + NOT high vol
            if kama_bear and below_mid and bear_breakout and vol_confirm and not in_high_vol:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                stop_long = entry_price - 2.5 * entry_atr
                if low[i] < stop_long:
                    desired_signal = 0.0
            elif position_side < 0:
                stop_short = entry_price + 2.5 * entry_atr
                if high[i] > stop_short:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (3 days) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Exit if KAMA flips
            if position_side > 0 and not kama_bull:
                desired_signal = 0.0
            if position_side < 0 and not kama_bear:
                desired_signal = 0.0
        
        # === TAKE PROFIT: Trail stop when 2R achieved ===
        if in_position and bars_held >= 3:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit > 2.0 * entry_atr:
                    # Trail: exit if price drops 1 ATR from highest
                    if low[i] < high[i] - 1.0 * atr_14[i]:
                        desired_signal = 0.0
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit > 2.0 * entry_atr:
                    if high[i] > low[i] + 1.0 * atr_14[i]:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals