#!/usr/bin/env python3
"""
Experiment #392: 12h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback v1

Hypothesis: Previous strategies failed due to overly complex regime detection
(ADX+Chop) that created too few trades. This version SIMPLIFIES to proven pattern:
- 1d KAMA for primary trend bias (adaptive, works better than HMA/EMA in crypto)
- 12h RSI pullback entries in trend direction (more frequent than breakouts)
- Fewer confluence requirements = more trades while maintaining quality

Why KAMA over HMA/EMA:
- KAMA adapts to volatility (fast in trends, slow in chop)
- Proven in crypto markets to reduce whipsaw
- Better risk-adjusted returns than static MAs

Entry Logic (SIMPLIFIED for trade frequency):
- Long: 1d KAMA bull + 12h KAMA bull + RSI(14) < 45 (pullback in uptrend)
- Short: 1d KAMA bear + 12h KAMA bear + RSI(14) > 55 (pullback in downtrend)
- Exit: RSI crosses mid (50) OR stoploss hit

Position sizing: 0.25 base, 0.30 when 1d+12h aligned strong
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=25 train, trades>=5 test, ALL symbols positive
Timeframe: 12h (targets 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_pullback_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - fast in trends, slow in chop
    period: efficiency ratio lookback
    fast: fast SC constant (default 2/3)
    slow: slow SC constant (default 2/31)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    sc = np.zeros(n)
    sc[:] = np.nan
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_slope(series, lookback=5):
    """Calculate slope of series over lookback period"""
    n = len(series)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(series[i]):
            x = np.arange(lookback)
            y = series[i-lookback+1:i+1]
            # Check for NaN in window
            if not np.any(np.isnan(y)):
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                if denominator > 1e-10:
                    slope[i] = numerator / denominator
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=21)
    kama_12h_fast = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate KAMA slope for trend strength
    kama_12h_slope = calculate_slope(kama_12h, lookback=5)
    kama_1d_slope = calculate_slope(kama_1d_aligned, lookback=5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d HTF TREND BIAS ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 12h KAMA TREND ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === KAMA SLOPE CONFIRMATION ===
        kama_12h_rising = not np.isnan(kama_12h_slope[i]) and kama_12h_slope[i] > 0
        kama_12h_falling = not np.isnan(kama_12h_slope[i]) and kama_12h_slope[i] < 0
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        # In uptrend: enter on RSI pullback to 35-45 zone
        # In downtrend: enter on RSI bounce to 55-65 zone
        rsi_pullback_long = rsi[i] < 45.0 and rsi[i] > 25.0
        rsi_pullback_short = rsi[i] > 55.0 and rsi[i] < 75.0
        
        # === SMA200 FILTER (optional confluence) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - trend + pullback) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 12h bull + RSI pullback
        if htf_bull and kama_bull:
            if rsi_pullback_long:
                # Strong signal if SMA200 also confirms
                if above_sma200 and kama_12h_rising:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + 12h bear + RSI pullback
        elif htf_bear and kama_bear:
            if rsi_pullback_short:
                # Strong signal if SMA200 also confirms
                if below_sma200 and kama_12h_falling:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC (RSI crosses mid or stoploss) ===
        # Exit long when RSI > 55 (overbought in uptrend)
        # Exit short when RSI < 45 (oversold in downtrend)
        if in_position and position_side > 0:
            if rsi[i] > 55.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi[i] < 45.0:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals