#!/usr/bin/env python3
"""
Experiment #975: 6h Primary + 12h/1d HTF — Simplified Trend + RSI Pullback

Hypothesis: Previous 6h strategies failed due to OVERLY STRICT entry conditions
(multiple regime filters, CRSI extremes, CHOP thresholds). This strategy uses
SIMPLIFIED logic to guarantee trades while maintaining HTF bias.

Key innovations:
1. LOOSE RSI thresholds (35/65 instead of 20/80) for MORE entries
2. Single HTF filter (1d HMA21) instead of multiple (1w + 1d + CHOP)
3. 12h momentum confirmation (simple ROC) for trend strength
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

Why this should work:
- Simpler conditions = MORE trades (avoid 0-trade failure mode)
- HTF bias still prevents counter-trend disasters
- RSI pullback in trend has proven edge across all crypto assets
- 6h captures multi-day swings without 4h noise or 12h lag

Entry conditions (LOOSE to guarantee 30+ trades/year):
- LONG = 1d HMA bull + 12h ROC > 0 + RSI(14) < 45 (pullback in uptrend)
- SHORT = 1d HMA bear + 12h ROC < 0 + RSI(14) > 55 (rally in downtrend)
- Exit = RSI crosses 50 opposite direction OR stoploss hit

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_simple_rsi_hma_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag vs traditional MA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_roc(close, period=12):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    roc_12h_raw = calculate_roc(df_12h['close'].values, period=12)
    roc_12h_aligned = align_htf_to_ltf(prices, df_12h, roc_12h_raw)
    
    # Calculate 6h indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    last_rsi = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(roc_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA + 12h ROC) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_12h_momentum_positive = roc_12h_aligned[i] > 0.0
        htf_12h_momentum_negative = roc_12h_aligned[i] < 0.0
        
        # === 6h RSI PULLBACK (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi_14[i] < 45  # Relaxed from 30
        rsi_overbought = rsi_14[i] > 55  # Relaxed from 70
        
        # RSI exit signals (cross 50)
        rsi_cross_below_50 = (last_rsi >= 50) and (rsi_14[i] < 50)
        rsi_cross_above_50 = (last_rsi <= 50) and (rsi_14[i] > 50)
        
        # === ENTRY LOGIC (SIMPLIFIED - FEWER FILTERS = MORE TRADES) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 12h momentum + RSI pullback
        if htf_1d_bull and htf_12h_momentum_positive:
            if rsi_oversold:
                desired_signal = SIZE_STRONG
            elif rsi_14[i] < 50 and in_position and position_side == 1:
                # Hold long if RSI still below 50
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + 12h momentum + RSI rally
        elif htf_1d_bear and htf_12h_momentum_negative:
            if rsi_overbought:
                desired_signal = -SIZE_STRONG
            elif rsi_14[i] > 50 and in_position and position_side == -1:
                # Hold short if RSI still above 50
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
        
        # === RSI EXIT (cross 50 opposite to position) ===
        if in_position and position_side > 0 and rsi_cross_below_50:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_cross_above_50:
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
        last_rsi = rsi_14[i]
    
    return signals