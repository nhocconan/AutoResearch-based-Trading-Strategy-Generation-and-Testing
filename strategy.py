#!/usr/bin/env python3
"""
Experiment #1347: 1d Primary + 4h HTF — Fisher Transform + Donchian Breakout

Hypothesis: Building on #1337 (Sharpe=0.618) which used 1d+1w with Donchian/HMA/RSI.
This experiment tests Fisher Transform for entry timing (better reversal detection in
bear/range markets per research) while keeping the proven Donchian breakout + HMA trend
structure. Using 4h HTF instead of 1w for more responsive trend signals.

Key design choices:
1. 4h HMA(21) for trend bias — faster than 1w, still smooth
2. Fisher Transform(9) for entry timing — catches reversals better than RSI
3. Donchian(20) breakout confirmation — ensures momentum alignment
4. ATR(14) trailing stop 2.5x — proven risk management from #1337
5. Position size 0.30 — conservative for daily volatility
6. NO regime filters — over-filtering caused 0-trade failures (#1335, #1338, #1340)
7. WIDE Fisher thresholds (-1.8/+1.8) — ensures sufficient trade frequency

Target: 25-45 trades/year on 1d, Sharpe > 0.618, trades >= 10 train, >= 3 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_donchian_hma_4h_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better for catching reversals in bear/range markets
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        # Normalize to 0-1 range
        normalized = (hl2 - lowest) / range_val
        
        # Clamp to avoid extreme values
        normalized = max(0.001, min(0.999, normalized))
        
        # Apply Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Apply EMA smoothing to Fisher value
        if i == period - 1:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
        else:
            alpha = 2.0 / (period + 1)
            fisher[i] = alpha * fisher_val + (1.0 - alpha) * fisher[i-1]
            fisher_prev[i] = fisher[i-1] if i > period - 1 else fisher_val
    
    return fisher, fisher_prev

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
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
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (WIDE thresholds for trade frequency) ===
        # Long: Fisher crosses above -1.8 (oversold reversal)
        fisher_long = fisher[i] > -1.8 and fisher_prev[i] <= -1.8
        # Short: Fisher crosses below +1.8 (overbought reversal)
        fisher_short = fisher[i] < 1.8 and fisher_prev[i] >= 1.8
        
        # Additional Fisher entry: extreme values with trend confirmation
        fisher_extreme_long = fisher[i] < -1.5 and trend_bull
        fisher_extreme_short = fisher[i] > 1.5 and trend_bear
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple paths to ensure trades happen
        if trend_bull:
            # Path 1: Fisher crossover from oversold (primary entry)
            if fisher_long:
                desired_signal = BASE_SIZE
            # Path 2: Fisher extreme + trend confirmation
            elif fisher_extreme_long:
                desired_signal = BASE_SIZE * 0.5
            # Path 3: Donchian breakout with trend (momentum entry)
            elif breakout_long:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY: Multiple paths to ensure trades happen
        elif trend_bear:
            # Path 1: Fisher crossover from overbought (primary entry)
            if fisher_short:
                desired_signal = -BASE_SIZE
            # Path 2: Fisher extreme + trend confirmation
            elif fisher_extreme_short:
                desired_signal = -BASE_SIZE * 0.5
            # Path 3: Donchian breakout with trend (momentum entry)
            elif breakout_short:
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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