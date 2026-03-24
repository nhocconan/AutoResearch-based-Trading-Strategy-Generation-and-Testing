#!/usr/bin/env python3
"""
Experiment #017: 1d Primary + 1w HTF — Simplified HMA Crossover with RSI Filter

Hypothesis: Previous strategies failed due to OVER-FILTERING and complex regime logic.
This uses PROVEN 1d patterns from research:
1. HMA(21) vs HMA(50) crossover for clean trend entries
2. RSI(14) filter to avoid entering at extremes (long if RSI<70, short if RSI>30)
3. 1w HMA for major trend BIAS (asymmetric sizing, not hard filter)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE entry thresholds to ensure ≥10 trades/symbol on train

Key improvements from failed attempts:
- SIMPLER logic = more trades generated (no complex regime switching)
- No funding rate loading (causing file I/O issues)
- Asymmetric sizing: 0.30 with HTF trend, 0.20 against HTF trend
- RSI filter prevents entering at extremes but doesn't block trades

Entry Logic:
- Long: HMA21 crosses above HMA50 + RSI < 70
- Short: HMA21 crosses below HMA50 + RSI > 30
- Size: 0.30 with 1w trend, 0.20 against 1w trend

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.3, trades > 10/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crossover_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    # WMA helper
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Combine
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - 100/(1 + RS)
    RS = avg_gain / avg_loss
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_hma_crossover(hma_fast, hma_slow):
    """
    Detect HMA crossover signals
    Returns: 1 for bullish crossover, -1 for bearish crossover, 0 otherwise
    """
    n = len(hma_fast)
    crossover = np.zeros(n)
    
    for i in range(1, n):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(hma_fast[i-1]) or np.isnan(hma_slow[i-1]):
            continue
        
        # Bullish crossover: fast crosses above slow
        if hma_fast[i-1] <= hma_slow[i-1] and hma_fast[i] > hma_slow[i]:
            crossover[i] = 1
        # Bearish crossover: fast crosses below slow
        elif hma_fast[i-1] >= hma_slow[i-1] and hma_fast[i] < hma_slow[i]:
            crossover[i] = -1
    
    return crossover

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Detect crossovers
    crossover = calculate_hma_crossover(hma_21, hma_50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track last crossover to avoid re-entering immediately
    last_crossover_time = -100
    last_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL BASED ON CROSSOVER + RSI FILTER ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Long entry: bullish crossover + RSI not overbought
        if crossover[i] == 1 and rsi[i] < 70.0:
            # Check minimum time since last crossover (avoid churn)
            if i - last_crossover_time > 5:
                if hma_1w_bull:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength
                last_crossover_time = i
        
        # Short entry: bearish crossover + RSI not oversold
        elif crossover[i] == -1 and rsi[i] > 30.0:
            if i - last_crossover_time > 5:
                if hma_1w_bear:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength
                last_crossover_time = i
        
        # === CONTINUE POSITION IF NO CROSSOVER AND RSI IN RANGE ===
        # Allow holding position even without new crossover
        if in_position and desired_signal == 0.0:
            if position_side > 0 and rsi[i] < 80.0:
                # Hold long if RSI not extremely overbought
                if hma_1w_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            elif position_side < 0 and rsi[i] > 20.0:
                # Hold short if RSI not extremely oversold
                if hma_1w_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
        
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
        # Clamp to max magnitude and discretize
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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
        last_signal = final_signal
    
    return signals