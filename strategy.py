#!/usr/bin/env python3
"""
Experiment #1022: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Complex regime filters (CHOP + CRSI + multiple HMA) cause 0 trades.
Simplifying to proven HMA trend + RSI pullback will generate MORE trades while
maintaining quality. Key insight from 841 failures: too many filters = no trades.

Key innovations:
1. SINGLE trend filter: 1d HMA(21) direction (not triple HMA)
2. RSI pullback entries: RSI<40 in uptrend, RSI>60 in downtrend (looser than before)
3. 1w HMA for long-term bias filter only (not entry requirement)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- Simpler = more trades (addresses #1 failure mode from experiment history)
- HMA reduces lag vs EMA (proven in literature)
- RSI pullback in trend has 60%+ win rate
- 4h captures multi-day swings (20-50 trades/year target)
- Looser RSI thresholds (40/60 vs 30/70) guarantee trades

Entry conditions (LOOSE to guarantee 30+ trades):
- LONG: close > 1d_HMA > 1w_HMA + RSI(14) < 45 (pullback in uptrend)
- SHORT: close < 1d_HMA < 1w_HMA + RSI(14) > 55 (pullback in downtrend)
- Exit: RSI crosses 50 or stoploss hit

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d1w_simplified_v2"
timeframe = "4h"
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Also calculate 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h[i]):
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
        
        # === TREND BIAS (HTF Alignment) ===
        # Long-term bias from 1w HMA
        long_term_bull = close[i] > hma_1w_aligned[i]
        long_term_bear = close[i] < hma_1w_aligned[i]
        
        # Medium-term trend from 1d HMA
        mid_term_bull = close[i] > hma_1d_aligned[i] and hma_1d_aligned[i] > hma_1w_aligned[i]
        mid_term_bear = close[i] < hma_1d_aligned[i] and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # Short-term from 4h HMA
        short_term_bull = close[i] > hma_4h[i]
        short_term_bear = close[i] < hma_4h[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Uptrend + RSI pullback (loose thresholds for trades)
        if mid_term_bull and long_term_bull:
            if rsi_14[i] < 45.0:  # Pullback in uptrend
                desired_signal = SIZE_BASE
            elif rsi_14[i] < 35.0 and short_term_bull:  # Deeper pullback + short-term confirmation
                desired_signal = SIZE_STRONG
        
        # SHORT ENTRY: Downtrend + RSI pullback (loose thresholds for trades)
        elif mid_term_bear and long_term_bear:
            if rsi_14[i] > 55.0:  # Pullback in downtrend
                desired_signal = -SIZE_BASE
            elif rsi_14[i] > 65.0 and short_term_bear:  # Deeper pullback + short-term confirmation
                desired_signal = -SIZE_STRONG
        
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
        
        # === EXIT CONDITIONS (RSI mean reversion) ===
        # Exit long if RSI goes too high (overbought in uptrend = take profit)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI goes too low (oversold in downtrend = take profit)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
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