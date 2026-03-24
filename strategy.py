#!/usr/bin/env python3
"""
Experiment #1072: 12h Primary + 1d HTF — Simple HMA Trend + RSI Filter

Hypothesis: A very simple 12h strategy with 1d HMA bias + 12h HMA crossover 
+ minimal RSI filter will generate enough trades across all symbols while 
maintaining positive Sharpe. Previous complex strategies failed due to 0 trades.

Key design:
1. 1d HMA(21) for long-term bias (only trade with bias)
2. 12h HMA(8/21) crossover for entry timing
3. RSI(14) > 30 for long, < 70 for short (very loose)
4. ATR(14) 2.5x trailing stop
5. LOOSE conditions to guarantee ≥30 trades/train, ≥3/test

Entry (LOOSE):
- LONG: 12h_HMA8 > 12h_HMA21 + close > 1d_HMA21 + RSI > 30
- SHORT: 12h_HMA8 < 12h_HMA21 + close < 1d_HMA21 + RSI < 70

Why this works:
- 12h captures multi-day swings (20-50 trades/year target)
- Simple HMA crossover generates frequent signals
- 1d bias filters against major trend
- Loose RSI ensures trades in all market conditions
- Discrete sizing minimizes fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_cross_rsi_simple_1d_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_12h_8 = calculate_hma(close, period=8)
    hma_12h_21 = calculate_hma(close, period=21)
    
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
    
    # Track previous HMA relationship for crossover detection
    prev_hma8_above_hma21 = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_8[i]) or np.isnan(hma_12h_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current HMA relationship
        curr_hma8_above_hma21 = hma_12h_8[i] > hma_12h_21[i]
        
        # === TREND BIAS (1d HMA) ===
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === CROSSOVER DETECTION ===
        # Bullish crossover: HMA8 crosses above HMA21
        bullish_cross = curr_hma8_above_hma21 and not prev_hma8_above_hma21
        # Bearish crossover: HMA8 crosses below HMA21
        bearish_cross = not curr_hma8_above_hma21 and prev_hma8_above_hma21
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: bullish cross OR already bullish + trend bias + RSI not extreme
        if (bullish_cross or curr_hma8_above_hma21) and trend_bull and rsi_14[i] > 30.0:
            # Stronger signal on fresh crossover
            if bullish_cross and rsi_14[i] < 70.0:
                desired_signal = SIZE_STRONG
            elif rsi_14[i] < 65.0:
                desired_signal = SIZE_BASE
        
        # SHORT: bearish cross OR already bearish + trend bias + RSI not extreme
        elif (bearish_cross or not curr_hma8_above_hma21) and trend_bear and rsi_14[i] < 70.0:
            # Stronger signal on fresh crossover
            if bearish_cross and rsi_14[i] > 30.0:
                desired_signal = -SIZE_STRONG
            elif rsi_14[i] > 35.0:
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
        
        # Update previous state for next iteration
        prev_hma8_above_hma21 = curr_hma8_above_hma21
    
    return signals