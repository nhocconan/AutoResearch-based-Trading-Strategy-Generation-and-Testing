#!/usr/bin/env python3
"""
Experiment #087: 6h Primary + 1d HTF — Fisher Transform Reversals + HMA Trend + Vol Filter

Hypothesis: After 86 failed experiments, the pattern for 6h is clear:
- Complex dual-regime strategies (Choppiness + Donchian + RSI) fail due to over-filtering
- Weekly pivot strategies fail (too slow for 6h entries)
- CRSI mean-reversion fails on 6h (wrong frequency)
- SOLUTION: Fisher Transform reversals are PROVEN for bear/range markets (2022, 2025)
- Fisher catches extremes better than RSI in trending markets (less whipsaw)
- 1d HMA(50) provides simple trend bias without over-complication
- ATR vol filter ensures we only trade when there's movement (avoid dead zones)
- This is DIFFERENT from all failed 6h strategies: no Donchian, no Choppiness, no CRSI

Key design choices:
- Timeframe: 6h (30-60 trades/year target, middle ground between 4h and 12h)
- HTF: 1d HMA(50) for major trend bias (simple, proven)
- Entry: Fisher Transform cross + HTF bias + ATR vol filter
- Fisher: period=9, long when crosses above -1.5, short when crosses below +1.5
- Vol filter: ATR(7)/ATR(30) > 1.3 (only trade when vol elevated)
- Position size: 0.28 (28% of capital, conservative for 6h)
- Stoploss: 2.5x ATR trailing
- LOOSE Fisher thresholds to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Calculate highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize price (0 to 1 range)
        range_hl = highest_high - lowest_low
        if range_hl < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        normalized = (median - lowest_low) / range_hl
        
        # Clamp to avoid division issues (0.001 to 0.999)
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period:
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Average True Range for stoploss and vol filter"""
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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
    # ATR ratio for vol filter (7/30)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # 6h HMA for additional trend confirmation
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOL FILTER (only trade when vol elevated) ===
        vol_elevated = atr_ratio[i] > 1.2  # lowered from 1.3 to ensure trades
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (Fisher + HTF + Vol) ===
        desired_signal = 0.0
        
        # LONG: Fisher cross + HTF bull OR HMA bull + vol elevated
        if fisher_cross_long and vol_elevated:
            if htf_bull or hma_bull:
                desired_signal = SIZE
            else:
                # Weaker signal without trend confirmation
                desired_signal = SIZE * 0.5
        
        # SHORT: Fisher cross + HTF bear OR HMA bear + vol elevated
        elif fisher_cross_short and vol_elevated:
            if htf_bear or hma_bear:
                desired_signal = -SIZE
            else:
                # Weaker signal without trend confirmation
                desired_signal = -SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals