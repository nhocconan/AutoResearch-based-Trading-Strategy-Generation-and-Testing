#!/usr/bin/env python3
"""
Experiment #1120: 6h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + RSI Filter

Hypothesis: The Ehlers Fisher Transform excels at catching reversals in bear/range markets
(like 2025 test period) while HMA provides trend direction. This combination should work
better than Choppiness/CRSI regime-switching which has failed repeatedly on 6h.

Key innovations:
1. Fisher Transform (period=9): Normalizes price to -1.5 to +1.5 range, catches reversals
   - Long when Fisher crosses above -1.5 from below (oversold reversal)
   - Short when Fisher crosses below +1.5 from above (overbought reversal)
2. 1d HMA(21) for intermediate trend direction (faster than SMA, less lag)
3. 1w HMA(21) for long-term bias filter (only trade with weekly trend)
4. RSI(14) confirmation: 35-65 range ensures we're not chasing extremes
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- 6h is middle ground: captures multi-day swings without 4h noise or 12h slowness
- Fisher Transform proven in literature for bear market reversals (Ehlers 2004)
- HMA reduces lag vs EMA/SMA, critical for 6h where bars are 6 hours apart
- 1d/1w HTF ensures we trade with higher timeframe trend (reduces whipsaws)
- Simpler entry logic than regime-switching (which caused 0 trades in exp 1111, 1117, 1119)
- Target: 30-60 trades/year on 6h timeframe

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher crosses above -1.5 + close > 1d_HMA + close > 1w_HMA*0.95 + RSI > 35
- SHORT: Fisher crosses below +1.5 + close < 1d_HMA + close < 1w_HMA*1.05 + RSI < 65

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_rsi_1d1w_v1"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Formula from "Cycle Analytics for Traders" (Ehlers 2004)
    
    Fisher = 0.5 * ln((1 + Value) / (1 - Value))
    where Value = 0.66 * PrevValue + 0.67 * (2 * (Close - LowN) / (HighN - LowN) - 1)
    
    Output range: typically -1.5 to +1.5
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate highest high and lowest low over period
    highest = np.full(n, np.nan, dtype=np.float64)
    lowest = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest[i] = np.max(close[i-period+1:i+1])
        lowest[i] = np.min(close[i-period+1:i+1])
    
    # Calculate Fisher Transform
    value = 0.0
    for i in range(period - 1, n):
        if highest[i] > lowest[i] and highest[i] > 1e-10:
            # Normalize price to -1 to +1 range
            price_norm = 2.0 * (close[i] - lowest[i]) / (highest[i] - lowest[i]) - 1.0
            
            # Apply smoothing with previous value
            value = 0.66 * value + 0.67 * price_norm
            
            # Clamp to avoid division by zero
            value = np.clip(value, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
            
            if i > period - 1:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

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
    
    for i in range(150, n):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i] * 0.95  # Small buffer
        hma_1w_bear = close[i] < hma_1w_aligned[i] * 1.05  # Small buffer
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Fisher crosses above -1.5 + bullish HTF + RSI not oversold
        if fisher_cross_up and hma_1d_bull and hma_1w_bull and rsi_14[i] > 35.0:
            # Stronger signal if RSI confirms momentum
            if rsi_14[i] > 45.0:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: Fisher crosses below +1.5 + bearish HTF + RSI not overbought
        elif fisher_cross_down and hma_1d_bear and hma_1w_bear and rsi_14[i] < 65.0:
            # Stronger signal if RSI confirms momentum
            if rsi_14[i] < 55.0:
                desired_signal = -SIZE_STRONG
            else:
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