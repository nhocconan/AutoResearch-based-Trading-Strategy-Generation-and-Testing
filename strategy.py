#!/usr/bin/env python3
"""
Experiment #1141: 4h Primary + 1d HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: After 830+ failed experiments, the key insight is that complex regime
switching (Choppiness + CRSI + multiple filters) causes 0 trades. The Ehlers Fisher
Transform is proven to catch reversals in bear/range markets (2022 crash, 2025 bear).

This strategy uses:
1. 1d HMA(21) for macro trend direction (simple, proven across BTC/ETH/SOL)
2. 4h Fisher Transform(14) for reversal entries (catches extremes in bear rallies)
3. LOOSE Fisher thresholds: cross above -1.5 for long, cross below +1.5 for short
4. ATR(14) 2.5x trailing stop (proven in research)
5. Position size 0.30 discrete (minimize fee churn)

Why this should beat Sharpe=0.612:
- Fisher Transform specifically designed for reversal capture in non-trending markets
- 2022 crash and 2025 bear are reversal-heavy environments (Fisher excels here)
- 1d HMA filter prevents counter-trend trades that destroyed returns
- Simple entry logic = more trades (target 30-50/year)
- No complex regime switching that caused 0 trades in exp #1130-#1140

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_1d_atr_reversal_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=14):
    """
    Ehlers Fisher Transform using (high + low) / 2 as price input.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    price = (high + low) / 2.0
    fisher_line = np.zeros(n)
    fisher_prev_line = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(price[i-period+1:i+1])
        lowest = np.min(price[i-period+1:i+1])
        
        if highest == lowest:
            fisher_line[i] = fisher_line[i-1] if i > period else 0.0
        else:
            normalized = (price[i] - lowest) / (highest - lowest)
            normalized = max(0.001, min(0.999, normalized))
            fisher_line[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        fisher_prev_line[i] = fisher_line[i-1] if i > period else 0.0
    
    fisher = fisher_line
    fisher_prev = fisher_prev_line
    return fisher, fisher_prev

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        # LOOSE thresholds to ensure trade frequency
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Also enter on deep extremes (no cross needed)
        fisher_deep_oversold = fisher[i] < -2.0
        fisher_deep_overbought = fisher[i] > 2.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + Fisher reversal (cross or deep oversold)
        if macro_bull and (fisher_cross_up or fisher_deep_oversold):
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + Fisher reversal (cross or deep overbought)
        elif macro_bear and (fisher_cross_down or fisher_deep_overbought):
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals