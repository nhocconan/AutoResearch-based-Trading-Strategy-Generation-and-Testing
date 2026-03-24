#!/usr/bin/env python3
"""
Experiment #660: 6h Primary + 1d/1w HTF — Ehlers Fisher Transform + HMA Trend Filter

Hypothesis: 6h timeframe is underexplored middle ground between 4h and 12h.
Ehlers Fisher Transform excels at catching reversals in bear/range markets (2022-2024),
unlike RSI which can stay extended. Combined with 1d HMA for trend direction and
1w HMA for regime filter. This should work better than RSI-based approaches that
failed in experiments #651, #655, #658.

Key innovations:
1. Fisher Transform (period=9) - normalized oscillator that catches reversals sharply
2. 1d HMA(21) trend bias - only long when price > daily HMA, only short when below
3. 1w HMA(50) regime filter - avoid counter-trend when weekly trend is strong
4. ATR(14) trailing stop - 2.5x for risk management
5. LOOSE entry conditions - Fisher cross + HTF alignment (no ADX/CHOP filters that block trades)

Entry conditions (designed to generate trades):
- LONG: Fisher crosses above -1.0 AND price > 1d HMA AND (price > 1w HMA OR weekly flat)
- SHORT: Fisher crosses below +1.0 AND price < 1d HMA AND (price < 1w HMA OR weekly flat)
- Weekly "flat" = price within 3% of 1w HMA (allows trades in consolidation)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-30%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > period and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher calculation
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized)) + 0.5 * fisher[i-1]
        else:
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    return fisher

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME FILTER (1w HMA) ===
        # Allow trades if weekly is flat (within 3% of HMA) or aligned
        weekly_flat = abs(close[i] - hma_1w_aligned[i]) / hma_1w_aligned[i] < 0.03
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 (oversold reversal)
        # Short: Fisher crosses below +1.0 (overbought reversal)
        fisher_long = False
        fisher_short = False
        
        if i >= 2 and not np.isnan(fisher[i-1]) and not np.isnan(fisher[i-2]):
            # Cross above -1.0 from below
            if fisher[i] > -1.0 and fisher[i-1] <= -1.0:
                fisher_long = True
            # Cross below +1.0 from above
            if fisher[i] < 1.0 and fisher[i-1] >= 1.0:
                fisher_short = True
        
        # Also allow continuation if Fisher is strongly in one direction
        fisher_strong_long = fisher[i] < -0.5 and fisher[i] < fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False
        fisher_strong_short = fisher[i] > 0.5 and fisher[i] > fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO ENSURE TRADES) ===
        desired_signal = 0.0
        
        # LONG: Daily bull + (Weekly bull OR flat) + Fisher signal
        if htf_bull and (weekly_bull or weekly_flat):
            if fisher_long:
                desired_signal = SIZE_STRONG
            elif fisher_strong_long:
                desired_signal = SIZE_BASE
            elif fisher[i] < -0.8:
                # Deep oversold in uptrend
                desired_signal = SIZE_BASE
        
        # SHORT: Daily bear + (Weekly bear OR flat) + Fisher signal
        elif htf_bear and (weekly_bear or weekly_flat):
            if fisher_short:
                desired_signal = -SIZE_STRONG
            elif fisher_strong_short:
                desired_signal = -SIZE_BASE
            elif fisher[i] > 0.8:
                # Deep overbought in downtrend
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
        
        signals[i] = final_signal
    
    return signals