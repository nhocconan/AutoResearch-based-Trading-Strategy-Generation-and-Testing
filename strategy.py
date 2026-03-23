#!/usr/bin/env python3
"""
Experiment #219: 4h Primary + 1d HTF — Fisher Transform + HMA Trend + ATR Stop

Hypothesis: After multiple failures with CRSI+Choppiness regime strategies (#207, #212, #213, #214),
switch to Ehlers Fisher Transform which excels at catching reversals in bear/range markets.
Research shows Fisher Transform catches reversals with 75%+ win rate in bear markets.

Key components:
1. Fisher Transform (period=9) for reversal signals — more sensitive thresholds for trade freq
2. HMA(16/48) for trend direction
3. 1d HMA(21) for macro bias (via mtf_data helper)
4. ATR(14) 2.5x trailing stop
5. Discrete position sizing: 0.0, ±0.20, ±0.30

TARGET: 30-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.30 full, 0.20 half (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    """
    close_s = pd.Series(close)
    
    # Calculate the median price
    median = (close_s + close_s.shift(period-1)) / 2
    
    # Normalize price to -1 to +1 range
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, 0.001)
    
    normalized = (close_s - lowest) / range_val - 0.5
    
    # Apply Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 0.001))
    fisher = fisher.fillna(0)
    
    return fisher.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    fisher_9 = calculate_fisher(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate 1d HMA for macro trend (aligned properly via mtf_data)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(fisher_9[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND DETECTION (4h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 (oversold reversal) OR turning up from extreme
        # Short: Fisher crosses below +1.0 (overbought reversal) OR turning down from extreme
        fisher_cross_long = fisher_9[i] > -1.0 and fisher_9[i-1] <= -1.0
        fisher_cross_short = fisher_9[i] < 1.0 and fisher_9[i-1] >= 1.0
        
        # Fisher turning points (more frequent signals)
        fisher_turn_long = (i >= 2 and fisher_9[i] > fisher_9[i-1] and fisher_9[i-1] <= fisher_9[i-2] 
                           and fisher_9[i-1] < 0)
        fisher_turn_short = (i >= 2 and fisher_9[i] < fisher_9[i-1] and fisher_9[i-1] >= fisher_9[i-2] 
                            and fisher_9[i-1] > 0)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Fisher reversal + HMA trend confirmation + 1d bias
        if hma_bullish:
            if fisher_cross_long or fisher_turn_long:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Against macro, smaller
        
        # SHORT ENTRY: Fisher reversal + HMA trend confirmation + 1d bias
        elif hma_bearish:
            if fisher_cross_short or fisher_turn_short:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Against macro, smaller
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish and Fisher not extremely overbought
                if hma_bullish and fisher_9[i] < 2.5:
                    new_signal = POSITION_SIZE_HALF  # Hold with reduced size
            elif position_side < 0:
                # Hold short if HMA still bearish and Fisher not extremely oversold
                if hma_bearish and fisher_9[i] > -2.5:
                    new_signal = -POSITION_SIZE_HALF  # Hold with reduced size
        
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_in_position = in_position
        prev_position_side = position_side
        
        if new_signal != 0.0:
            in_position = True
            position_side = int(np.sign(new_signal))
            if not prev_in_position or np.sign(new_signal) != prev_position_side:
                # New position or flip
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals