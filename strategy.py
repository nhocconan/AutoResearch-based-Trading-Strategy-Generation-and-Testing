#!/usr/bin/env python3
"""
Experiment #123: 1d Primary + 1w HTF — Simplified Donchian Breakout with Macro Trend

Hypothesis: Previous strategies (#117, #121, #122) failed due to over-complication with
regime switching and too many exit conditions. This simplifies to core proven elements:

1) 1w HMA(21) for macro trend bias — only trade breakouts in trend direction
2) 1d Donchian(20) breakout — clean trend-following entry
3) 1d HMA(21/50) crossover for intermediate trend confirmation
4) Volume confirmation — breakout volume > 1.5x 20-day avg
5) ATR(14) trailing stop at 2.5x — simple, effective risk management
6) Exit on opposite Donchian break OR 1w trend reversal

Key improvements over #122:
- Pre-calculate ALL indicators before loop (RSI was calculated inside loop = bug)
- Simpler hold logic (just maintain signal, don't over-check conditions)
- Fewer exit conditions (remove RSI extreme exits that cut winners early)
- Cleaner position tracking

Position size: 0.25 base, 0.30 with volume confluence
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_simple_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength) - pre-calculated
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate ALL 1d indicators BEFORE loop (Rule 8 - Performance)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_slope_positive = hma_1w_slope[i] > 0.3
        hma_slope_negative = hma_1w_slope[i] < -0.3
        
        # === 1d TREND FILTER ===
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1]
        prev_low = donchian_lower[i-1]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.3
        volume_strong = volume_ratio > 1.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1w trend up + 1d trend up + Donchian breakout + volume
        if price_above_hma_1w and hma_1d_bullish and breakout_long:
            if volume_confirmed:
                new_signal = POSITION_SIZE_BASE
                if volume_strong and hma_slope_positive:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 1w trend down + 1d trend down + Donchian breakout + volume
        if price_below_hma_1w and hma_1d_bearish and breakout_short:
            if volume_confirmed:
                new_signal = -POSITION_SIZE_BASE
                if volume_strong and hma_slope_negative:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain signal unless exit conditions met
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit if 1w trend reverses strongly negative
            if price_below_hma_1w and hma_slope_negative:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if 1w trend reverses strongly positive
            if price_above_hma_1w and hma_slope_positive:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_long:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals