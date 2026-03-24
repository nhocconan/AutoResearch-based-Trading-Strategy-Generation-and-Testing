#!/usr/bin/env python3
"""
Experiment #800: 6h Primary + 1d/1w HTF — Fisher Transform Reversal Strategy

Hypothesis: 6h timeframe with Fisher Transform entries captures reversals better than RSI
in bear/range markets (2025+ test period). Fisher Transform normalizes price to Gaussian
distribution, providing clearer reversal signals at extremes. Combined with 1d/1w HMA
trend filters for directional bias.

Key innovations:
1. Fisher Transform (period=9) for entry timing — proven in bear markets
2. 1w HMA(21) for major trend bias (avoid counter-trend trades)
3. 1d HMA(21) for intermediate trend confirmation
4. 6h Fisher < -1.5 for long entries, > +1.5 for short entries
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (balanced for trade generation):
- LONG: 1w HMA bull + 1d HMA bull + Fisher < -1.0 (loosened from -1.5)
- SHORT: 1w HMA bear + 1d HMA bear + Fisher > +1.0 (loosened from +1.5)
- Exit: Fisher crosses back through 0 OR stoploss hit

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals at extremes better than RSI in bear markets
    Reference: Ehlers, J.F. (2002) "Fishing with a Fisher Transform"
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize price to range -1 to +1
    fisher_input = np.zeros(n)
    fisher_input[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            normalized = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
            # Clamp to avoid division issues
            normalized = max(-0.999, min(0.999, normalized))
            fisher_input[i] = normalized
    
    # Apply Fisher Transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(fisher_input[i]) and abs(fisher_input[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + fisher_input[i]) / (1.0 - fisher_input[i]))
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher_9 = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Fisher exit tracking
    fisher_entry_value = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher_9[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA - Major Trend) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF BIAS (1d HMA - Intermediate Trend) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long entry: Fisher < -1.0 (oversold, loosened from -1.5 for more trades)
        # Short entry: Fisher > +1.0 (overbought, loosened from +1.5 for more trades)
        fisher_oversold = fisher_9[i] < -1.0
        fisher_overbought = fisher_9[i] > 1.0
        
        # Fisher exit: crosses back through 0
        fisher_exit_long = False
        fisher_exit_short = False
        
        if i > 0 and not np.isnan(fisher_9[i-1]):
            # Long exit: Fisher was negative, now crosses above 0
            fisher_exit_long = (fisher_9[i-1] < 0.0) and (fisher_9[i] >= 0.0)
            # Short exit: Fisher was positive, now crosses below 0
            fisher_exit_short = (fisher_9[i-1] > 0.0) and (fisher_9[i] <= 0.0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d bull + Fisher oversold
        if htf_1w_bull and htf_1d_bull and fisher_oversold:
            if fisher_9[i] < -1.5:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + 1d bear + Fisher overbought
        elif htf_1w_bear and htf_1d_bear and fisher_overbought:
            if fisher_9[i] > 1.5:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
            # Fisher exit for long
            if fisher_exit_long:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
            # Fisher exit for short
            if fisher_exit_short:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                fisher_entry_value = fisher_9[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                fisher_entry_value = 0.0
        
        signals[i] = final_signal
    
    return signals