#!/usr/bin/env python3
"""
Experiment #1184: 12h Primary + 1d/1w HTF — HMA Trend + Fisher Transform Reversal

Hypothesis: After analyzing 960+ failures, the key insight is that RSI pullback entries
work but can be improved with Fisher Transform for better reversal timing. Fisher Transform
normalizes price to Gaussian distribution, making extreme readings (-2 to +2) more reliable
than RSI for catching turning points in pullbacks.

Strategy Logic:
1. Daily HMA(21) = primary trend filter (price above = long bias, below = short bias)
2. Weekly HMA(21) = trend strength confirmation (increases position size when aligned)
3. 12h Fisher Transform(9) = entry timing (cross above -1.5 = long, cross below +1.5 = short)
4. ATR(14) 2.0x trailing stop = risk management

Why Fisher over RSI:
- Fisher Transform has sharper signals at extremes (±1.5 vs RSI 30/70)
- Better at catching reversals in bear market rallies (critical for 2025 test period)
- Less whipsaw in ranging markets compared to RSI

Entry Logic (LOOSE to guarantee trades):
- LONG: price > 1d_HMA AND Fisher crosses above -1.5 (from below)
- SHORT: price < 1d_HMA AND Fisher crosses below +1.5 (from above)
- No additional filters (learned: over-filtering = 0 trades)

Position Sizing:
- Base: 0.25 (25% of capital)
- Strong (1w HMA aligned): 0.30 (30% of capital)
- Discrete levels only to minimize fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_fisher_reversal_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Reference: "Cybernetic Analysis for Stocks and Futures" by John Ehlers
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using highest high / lowest low over period
    3. Apply Fisher transform: 0.5 * ln((1 + x) / (1 - x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1
    normalized = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        if highest > lowest:
            normalized[i] = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
            # Clamp to avoid division by zero in Fisher
            normalized[i] = max(-0.999, min(0.999, normalized[i]))
    
    # Fisher transform
    fisher_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(normalized[i]):
            fisher_raw[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher_9 = calculate_fisher_transform(high, low, close, period=9)
    
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
    
    # Track Fisher for crossover detection
    prev_fisher = np.nan
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        
        if np.isnan(fisher_9[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_9[i] if not np.isnan(fisher_9[i]) else prev_fisher
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for additional confirmation (not required)
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM CROSSOVER DETECTION ===
        fisher = fisher_9[i]
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher):
            # Long signal: Fisher crosses ABOVE -1.5 (from below)
            if prev_fisher <= -1.5 and fisher > -1.5:
                fisher_cross_long = True
            # Short signal: Fisher crosses BELOW +1.5 (from above)
            if prev_fisher >= 1.5 and fisher < 1.5:
                fisher_cross_short = True
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Price above 1d HMA + Fisher crosses above -1.5
        if price_above_1d and fisher_cross_long:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic uptrend
        
        # SHORT: Price below 1d HMA + Fisher crosses below +1.5
        elif price_below_1d and fisher_cross_short:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic downtrend
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
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
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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
        
        # Update previous Fisher for next iteration
        prev_fisher = fisher
    
    return signals