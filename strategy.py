#!/usr/bin/env python3
"""
Experiment #1075: 6h Primary + 12h/1d HTF — Ehlers Fisher Transform + HMA Trend + RSI Confirmation

Hypothesis: The 6h timeframe captures multi-day cycles with less noise than 4h. Using Ehlers'
Fisher Transform for precise reversal timing, combined with 12h HMA for trend bias and RSI
for momentum confirmation, should reduce whipsaws that destroyed previous 6h strategies.

Key innovations:
1. Fisher Transform (period=9) - normalizes price to clear reversal signals at extremes
2. 12h HMA(21) for intermediate trend - faster than 1d, slower than 6h
3. 1d HMA(21) for long-term bias - filters counter-trend trades
4. RSI(14) momentum confirmation - avoids entering at weak reversals
5. LOOSE Fisher levels (-1.5/+1.5) to guarantee trades (learned from 0-trade failures)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- Fisher Transform catches reversals in bear/range markets (2022-2023, 2025+)
- 12h HMA filters direction without being too slow (1d) or too fast (6h)
- 6h = 4 bars/day captures multi-day swings without 4h noise
- LOOSE entry conditions ensure trades on all symbols (BTC/ETH/SOL)

Entry conditions (LOOSE to guarantee >=30 trades):
- LONG: Fisher < -1.5 + price > 12h_HMA + RSI > 35 + 1d_HMA not strongly bearish
- SHORT: Fisher > +1.5 + price < 12h_HMA + RSI < 65 + 1d_HMA not strongly bullish

Target: Sharpe>0.45 (beat current 6h best of 0.424), trades>=30 train, trades>=5 test
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_rsi_12h1d_v1"
timeframe = "6h"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_fisher(close, period=9):
    """
    Fisher Transform - John Ehlers
    Normalizes price to Gaussian distribution for clear reversal signals
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > period and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher_prev[i-1] if i > period and not np.isnan(fisher_prev[i-1]) else 0.0
            continue
        
        normalized = 2.0 * (close[i] - lowest) / range_val - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth Fisher
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_raw + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_raw
            fisher_prev[i] = fisher_raw
    
    return fisher, fisher_prev

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_prev = calculate_fisher(close, period=9)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h + 1d HMA) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_12h_bull and hma_1d_bull
        strong_bear = hma_12h_bear and hma_1d_bear
        
        # === ENTRY LOGIC (Fisher Transform + RSI + HMA) ===
        desired_signal = 0.0
        
        # LONG: Fisher oversold + 12h bullish + RSI not too weak
        # LOOSE conditions to ensure trades
        if fisher[i] < -1.5 and hma_12h_bull and rsi_14[i] > 30:
            if strong_bull:  # 1d also bullish = stronger signal
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # Also allow long on Fisher crossover from very oversold
        elif fisher_prev[i] < -2.0 and fisher[i] > fisher_prev[i] and hma_12h_bull and rsi_14[i] > 35:
            desired_signal = SIZE_BASE
        
        # SHORT: Fisher overbought + 12h bearish + RSI not too strong
        # LOOSE conditions to ensure trades
        elif fisher[i] > 1.5 and hma_12h_bear and rsi_14[i] < 70:
            if strong_bear:  # 1d also bearish = stronger signal
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # Also allow short on Fisher crossover from very overbought
        elif fisher_prev[i] > 2.0 and fisher[i] < fisher_prev[i] and hma_12h_bear and rsi_14[i] < 65:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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