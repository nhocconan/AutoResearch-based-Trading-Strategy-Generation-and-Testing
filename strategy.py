#!/usr/bin/env python3
"""
Experiment #1453: 1d Primary + 1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Daily timeframe with weekly HTF trend filter using Ehlers Fisher Transform
for entry timing will outperform RSI-based strategies in bear/range markets (2025 test).

Why this approach:
1. Fisher Transform (Ehlers) normalizes price to Gaussian distribution, better at catching
   reversals than RSI in choppy/bear markets — proven in research literature
2. KAMA (Kaufman Adaptive) automatically adjusts smoothing based on volatility —
   less whipsaw than EMA/HMA in range markets like 2025
3. 1w HMA provides macro trend filter without over-complicating
4. Simpler logic = fewer false signals, target 25-40 trades/year
5. Asymmetric entries: only long when price>1w_HMA, only short when price<1w_HMA

Key differences from current best (mtf_1d_donchian_hma_rsi_1w_atr_v1):
- Fisher Transform(9) instead of RSI(14) for entry timing
- KAMA(21) instead of HMA on primary timeframe
- No Donchian breakout, no Choppiness Index — cleaner signal logic
- Focus on reversal entries within macro trend direction

Target: Sharpe > 0.618 (beat current best), trades >= 30 train, >= 5 test, DD < -35%
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Automatically adjusts smoothing based on market volatility/noise
    Better than EMA in choppy/range markets
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for better reversal detection
    Returns Fisher value and signal line (1-period lag)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest > lowest:
            # Normalize price to 0-1 range
            value = (2.0 * (high[i] + low[i]) / 2.0 - (highest + lowest)) / (highest - lowest)
            # Clamp to avoid division issues
            value = max(-0.999, min(0.999, value))
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
            
            if i > 0 and not np.isnan(fisher[i - 1]):
                fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_21 = calculate_kama(close, period=21)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_21[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        # === MACRO TREND (1w HMA) - strongest filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bull = close[i] > kama_21[i]
        kama_bear = close[i] < kama_21[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = (fisher[i] > -1.5 and fisher_signal[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = (fisher[i] < 1.5 and fisher_signal[i] >= 1.5)
        
        # Fisher extreme levels for additional confirmation
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === DESIRED SIGNAL - FISHER + KAMA + MACRO TREND ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Macro bull + Fisher reversal from oversold + KAMA bull
        if macro_bull and fisher_long and kama_bull:
            desired_signal = BASE_SIZE
        # Path 2: Macro bull + Fisher extreme oversold (stronger signal)
        elif macro_bull and fisher_oversold and kama_bull:
            desired_signal = BASE_SIZE
        # Path 3: Macro bull + KAMA bull (trend continuation)
        elif macro_bull and kama_bull and fisher[i] > fisher_signal[i]:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        # Path 1: Macro bear + Fisher reversal from overbought + KAMA bear
        elif macro_bear and fisher_short and kama_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Macro bear + Fisher extreme overbought (stronger signal)
        elif macro_bear and fisher_overbought and kama_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Macro bear + KAMA bear (trend continuation)
        elif macro_bear and kama_bear and fisher[i] < fisher_signal[i]:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE
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