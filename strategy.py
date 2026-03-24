#!/usr/bin/env python3
"""
Experiment #180: 6h Primary + 1d/1w HTF — Fisher Transform Reversals with Dual HTF Filter

Hypothesis: 6h timeframe captures multi-day swings without excessive noise. Fisher Transform
(excursions from Gaussian distribution) catches reversals better than RSI in bear/range markets.
Using BOTH 1d AND 1w HMA alignment ensures we only trade in direction of major trend, avoiding
whipsaws like 2022 crash. This differs from failed 6h strategies by:
1. Using Fisher Transform instead of RSI/CRSI (better reversal detection)
2. Requiring BOTH 1d AND 1w HTF alignment (stricter than single HTF)
3. Mean reversion focus (works in 2025 bear/range market)
4. Simple discrete signals to minimize fee churn

Entry: Fisher crosses -1.5 (long) or +1.5 (short) + both HTF agree
Exit: Fisher crosses 0 opposite direction OR 2.5x ATR stoploss
Size: 0.25 base, 0.30 for strong HTF alignment

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_reversal_1d1w_v1"
timeframe = "6h"
leverage = 1.0

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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution
    Catches turning points better than RSI in bear/range markets
    
    Steps:
    1. Calculate typical price = (high + low) / 2
    2. Normalize to range -1 to +1 over lookback period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 range over period
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize typical price to -0.99 to +0.99 (avoid division by zero)
            normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            fisher_raw[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    
    # 6h HMA for local trend confirmation
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher signal tracking (to avoid repeated entries)
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(hma_6h[i]):
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
        
        # === HTF BIAS (Require BOTH 1d AND 1w alignment) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both HTF agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # Weak bias: only 1d agrees (1w neutral or opposite)
        htf_weak_bull = htf_1d_bull and not htf_1w_bear
        htf_weak_bear = htf_1d_bear and not htf_1w_bull
        
        # === 6h HMA LOCAL TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_signal = False
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_long_signal = (fisher[i-1] < -1.5 and fisher[i] >= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_signal = False
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_short_signal = (fisher[i-1] > 1.5 and fisher[i] <= 1.5)
        
        # Exit signals: Fisher crosses 0 in opposite direction
        fisher_exit_long = False
        fisher_exit_short = False
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_exit_long = (fisher[i-1] > 0 and fisher[i] <= 0)  # Long exit
            fisher_exit_short = (fisher[i-1] < 0 and fisher[i] >= 0)  # Short exit
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Fisher oversold reversal + HTF bull bias + 6h HMA confirmation
        if fisher_long_signal and not prev_fisher_long_signal:
            if htf_strong_bull and hma_6h_bull:
                desired_signal = SIZE_STRONG
            elif htf_weak_bull and hma_6h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: Fisher overbought reversal + HTF bear bias + 6h HMA confirmation
        if fisher_short_signal and not prev_fisher_short_signal:
            if htf_strong_bear and hma_6h_bear:
                desired_signal = -SIZE_STRONG
            elif htf_weak_bear and hma_6h_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
            # Also exit on Fisher cross below 0
            if fisher_exit_long:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
            # Also exit on Fisher cross above 0
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                # Reset Fisher signal flags to prevent repeated entries
                prev_fisher_long_signal = False
                prev_fisher_short_signal = False
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                prev_fisher_long_signal = False
                prev_fisher_short_signal = False
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
        
        # Update Fisher signal flags for next iteration
        prev_fisher_long_signal = fisher_long_signal
        prev_fisher_short_signal = fisher_short_signal
        
        signals[i] = final_signal
    
    return signals