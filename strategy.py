#!/usr/bin/env python3
"""
Experiment #473: 1d Primary + 1w HTF — Fisher Transform + BB Squeeze + Asymmetric Regime

Hypothesis: Based on research showing Ehlers Fisher Transform catches reversals better than RSI
in bear/range markets (2022 crash, 2025 bear). Combined with Bollinger Band Width squeeze
detection for breakout timing, and 1w HMA for HTF trend bias.

Key innovations:
1. Fisher Transform (period=9) - asymmetric thresholds: long when crosses above -1.5, short when crosses below +1.5
2. BB Width percentile (30d) < 20% = volatility squeeze (pre-breakout signal)
3. Asymmetric entry logic: longs only when price > 1w HMA, shorts only when price < 1w HMA
4. 1d primary timeframe = 20-50 trades/year target (fee-efficient)
5. ATR(14) trailing stop at 2.0x for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: Fisher Transform normalizes price into bounded oscillator that catches
reversals at extremes. BB squeeze identifies low-volatility periods before breakouts.
1w HMA filter prevents counter-trend trades (major source of losses in 2022).
Asymmetric logic acknowledges bear market bias - harder to go long, easier to short.
Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_bb_squeeze_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price into bounded oscillator.
    Formula: Fisher = 0.5 * ln((1 + EValue) / (1 - EValue))
    where EValue = 0.33 * 2 * ((close - low5) / (high5 - low5)) - 0.33
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            continue
        
        # EValue calculation
        evalue = 0.33 * 2.0 * ((close[i] - lowest) / (highest - lowest) - 0.5)
        evalue = np.clip(evalue, -0.99, 0.99)  # Prevent ln domain error
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1.0 + evalue) / (1.0 - evalue + 1e-10))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width and its 30-day percentile."""
    n = len(close)
    bb_width = np.full(n, np.nan)
    bb_width_pct = np.full(n, np.nan)
    
    # Calculate BB width
    for i in range(period, n):
        window = close[i-period+1:i+1]
        sma = np.mean(window)
        std = np.std(window)
        
        if sma > 1e-10:
            bb_width[i] = (2.0 * std_mult * std) / sma
    
    # Calculate 30-day percentile of BB width
    lookback = 30
    for i in range(period + lookback, n):
        if np.isnan(bb_width[i]):
            continue
        window = bb_width[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_width_pct[i] = np.sum(valid[:-1] < bb_width[i]) / (len(valid) - 1) * 100
    
    return bb_width, bb_width_pct

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if period < 2:
        return hma
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i-span+1:i+1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_input = 2.0 * wma_half - wma_full
    
    hma = wma(hma_input, sqrt_period)
    
    return hma

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    fisher, fisher_prev = calculate_fisher_transform(close, period=9)
    bb_width, bb_width_pct = calculate_bb_width(close, period=20, std_mult=2.0)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align 1w HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        if np.isnan(bb_width_pct[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY SQUEEZE DETECTION ===
        is_squeeze = bb_width_pct[i] < 25.0  # BB width in bottom 25% of 30-day range
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme oversold/overbought
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (only when price > 1w HMA - bullish bias)
        if price_above_hma_1w:
            long_score = 0
            
            # Fisher reversal signal
            if fisher_cross_long:
                long_score += 3
            elif fisher_extreme_long:
                long_score += 2
            
            # Squeeze breakout bonus
            if is_squeeze:
                long_score += 1
            
            # Strong 1w trend bonus
            if close[i] > hma_1w_aligned[i] * 1.02:  # 2% above 1w HMA
                long_score += 1
            
            if long_score >= 3:
                desired_signal = SIZE_LONG
        
        # SHORT ENTRIES (only when price < 1w HMA - bearish bias)
        if desired_signal == 0.0 and price_below_hma_1w:
            short_score = 0
            
            # Fisher reversal signal
            if fisher_cross_short:
                short_score += 3
            elif fisher_extreme_short:
                short_score += 2
            
            # Squeeze breakdown bonus
            if is_squeeze:
                short_score += 1
            
            # Strong 1w downtrend bonus
            if close[i] < hma_1w_aligned[i] * 0.98:  # 2% below 1w HMA
                short_score += 1
            
            if short_score >= 3:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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