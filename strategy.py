#!/usr/bin/env python3
"""
Experiment #1674: 4h Primary + 12h/1d HTF — Donchian Breakout with HMA Trend Filter

Hypothesis: Previous 4h failures over-complicated entry logic with too many filters.
The BEST strategy (1d Donchian + HMA + RSI + 1w, Sharpe=0.618) proves Donchian breakouts
work on higher timeframes. This adapts that proven pattern to 4h with:

1. Donchian(20) breakout as PRIMARY signal (proven on 1d best strategy)
2. HMA(21) on 12h for trend bias (not hard filter - just probability weight)
3. RSI(14) for entry timing confirmation (RSI > 50 for long breakouts)
4. Simpler regime: just price vs HMA, no Choppiness complexity
5. LOOSE entry conditions to ensure 30+ trades/symbol (learned from 0-trade failures)

Key differences from failed 4h attempts:
- Donchian breakout instead of CRSI mean reversion (CRSI failed on 4h)
- Single HTF (12h HMA) instead of dual HTF complexity
- RSI as confirmation not filter (RSI > 45 not > 60)
- ATR stop at 3.0x (wider to avoid whipsaw exits)
- Size: 0.25 base, 0.30 with HTF trend confirmation

Entry Logic:
- Long: Price breaks Donchian(20) high + RSI > 45 + (optional: above 12h HMA)
- Short: Price breaks Donchian(20) low + RSI < 55 + (optional: below 12h HMA)
- Size: 0.30 with 12h HMA trend, 0.25 against

Risk: 3.0x ATR trailing stop, discrete signal levels (0.0, ±0.25, ±0.30)
Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_12h_atr_v1"
timeframe = "4h"
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
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over N periods
    Lower = lowest low over N periods
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_avg[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_avg[i-1] / loss_avg[i-1]))
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track previous Donchian levels for breakout detection
    prev_upper = 0.0
    prev_lower = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === BREAKOUT DETECTION ===
        # Long breakout: price crosses ABOVE Donchian upper
        long_breakout = (close[i] > donchian_upper[i]) and (prev_upper > 0 and close[i-1] <= prev_upper)
        
        # Short breakout: price crosses BELOW Donchian lower
        short_breakout = (close[i] < donchian_lower[i]) and (prev_lower > 0 and close[i-1] >= prev_lower)
        
        # === HTF TREND BIAS ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === RSI CONFIRMATION (LOOSE thresholds for trade generation) ===
        rsi_ok_long = rsi[i] > 45.0  # Not oversold
        rsi_ok_short = rsi[i] < 55.0  # Not overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if long_breakout and rsi_ok_long:
            if hma_12h_bull:
                desired_signal = TREND_SIZE  # 0.30 with trend
            else:
                desired_signal = BASE_SIZE  # 0.25 against trend
        elif short_breakout and rsi_ok_short:
            if hma_12h_bear:
                desired_signal = -TREND_SIZE  # -0.30 with trend
            else:
                desired_signal = -BASE_SIZE  # -0.25 against trend
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= TREND_SIZE * 0.9:
            final_signal = TREND_SIZE
        elif desired_signal <= -TREND_SIZE * 0.9:
            final_signal = -TREND_SIZE
        elif desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.9:
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
        
        # Update previous Donchian levels
        prev_upper = donchian_upper[i]
        prev_lower = donchian_lower[i]
    
    return signals