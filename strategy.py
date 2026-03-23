#!/usr/bin/env python3
"""
Experiment #129: 4h Primary + 1d HTF — Simplified Donchian Breakout with HMA Trend

Hypothesis: Recent failures (#117-#128) show complex regime switching adds lag and reduces trades.
The current best (mtf_4h_crsi_chop_donchian_regime_1d1w_v3, Sharpe=0.486) uses 4h with 1d/1w HTF.
This simplifies to pure trend-following:

1) 1d HMA(21) for macro trend bias — only trade breakouts in trend direction
2) 4h Donchian(20) breakout — price breaks 20-period high/low for entry
3) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown
4) Simple exit: opposite Donchian break OR 1d trend reversal

Why this should work:
- 4h produces 25-50 trades/year (optimal for fee drag)
- 1d HTF is faster than 1w, catches trend changes sooner
- Removed volume filter (killed too many valid signals)
- Removed RSI take-profit (exited winners too early)
- Discrete position sizes (0.0, ±0.25, ±0.30) minimize fee churn

Position size: 0.25 base, 0.30 with strong HTF confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_simple_1d_v1"
timeframe = "4h"
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
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.3
        hma_slope_negative = hma_1d_slope[i] < -0.3
        
        # === 4h TREND FILTER ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1d trend up + 4h trend up + Donchian breakout
        if price_above_hma_1d and hma_4h_bullish and breakout_long:
            new_signal = POSITION_SIZE_BASE
            if hma_slope_positive:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 1d trend down + 4h trend down + Donchian breakout
        if price_below_hma_1d and hma_4h_bearish and breakout_short:
            new_signal = -POSITION_SIZE_BASE
            if hma_slope_negative:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in trend and not stopped out
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if above Donchian mid and 1d trend intact
                if close[i] > donchian_mid[i] and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if below Donchian mid and 1d trend intact
                if close[i] < donchian_mid[i] and price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit if 1d trend reverses down
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if 1d trend reverses up
            if price_above_hma_1d and hma_slope_positive:
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
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals