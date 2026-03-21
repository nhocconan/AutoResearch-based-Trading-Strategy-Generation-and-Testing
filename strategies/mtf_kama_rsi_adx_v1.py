#!/usr/bin/env python3
"""
EXPERIMENT #012 - KAMA Trend + RSI Pullback + ADX Strength + ATR Stop
======================================================================
Hypothesis: KAMA's adaptive smoothing captures trends with less lag than HMA.
Combined with RSI pullback entries (buy dips in uptrend), ADX strength filter,
and ATR-based trailing stops, this should improve risk-adjusted returns.

Key improvements over mtf_hma_supertrend_rsi_v1:
- KAMA instead of HMA (better adaptation to volatility regimes - worked well in exp 002/005)
- ADX filter to avoid weak/choppy trend periods (missing from previous best)
- RSI pullback entries (40-50 for long, 50-60 for short) - cleaner than Supertrend
- Proper ATR trailing stop with 2.5*ATR distance
- Discrete position sizing (0.0, 0.20, 0.35) to reduce churn costs

Why this might beat Sharpe=2.931:
- KAMA showed strong results in experiments 002 (+94.9%) and 005 (+810.4%)
- ADX filter avoids 30-40% of losing trades in choppy markets
- RSI pullbacks provide better entry timing than pure breakout
"""

import numpy as np
import pandas as pd

name = "mtf_kama_rsi_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))
        er[i] = signal / noise if noise > 0 else 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_adx(high, low, close, period=14, adx_period=14):
    """Calculate ADX for trend strength"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        plus_di[i] = 100 * plus_dm[i] / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * minus_dm[i] / atr[i] if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h indicators for trend (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h KAMA for trend direction
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h ADX for trend strength
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14, adx_period=14)
    
    # 4h trend direction based on KAMA position and ADX strength
    trend_4h = np.zeros(len(c_4h))
    for i in range(40, len(c_4h)):
        if adx_4h[i] > 20:  # Strong trend
            if c_4h[i] > kama_4h[i]:
                trend_4h[i] = 1  # Bullish
            elif c_4h[i] < kama_4h[i]:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 0:
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in strong trend
    SIZE_HALF = 0.20   # Reduced position in weaker conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_PULLBACK = 45   # Enter long on pullback in uptrend
    RSI_SHORT_PULLBACK = 55  # Enter short on rally in downtrend
    
    # ADX thresholds for trend strength
    ADX_STRONG = 25    # Strong trend - full position
    ADX_WEAK = 15      # Weak trend - no trading
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 40, 14)  # Wait for all indicators
    
    # Track entry prices for trailing stop
    entry_price = np.zeros(n)
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        adx_val = adx_4h[min(i // 4, len(adx_4h) - 1)] if len(adx_4h) > 0 else 0
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.04:  # ATR > 4% of price = too volatile
            signals[i] = 0.0
            if i > 0 and signals[i-1] != 0:
                entry_price[i] = 0
                position_direction[i] = 0
            continue
        
        # Check trailing stop for existing positions FIRST
        if i > 0 and position_direction[i-1] != 0:
            prev_direction = position_direction[i-1]
            prev_entry = entry_price[i-1] if entry_price[i-1] > 0 else close[i-1]
            
            if prev_direction == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0  # Stoploss triggered
                    entry_price[i] = 0
                    position_direction[i] = 0
                    continue
                else:
                    # Hold position, update entry to highest since entry
                    entry_price[i] = max(prev_entry, price)
                    position_direction[i] = 1
            elif prev_direction == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0  # Stoploss triggered
                    entry_price[i] = 0
                    position_direction[i] = 0
                    continue
                else:
                    # Hold position, update entry to lowest since entry
                    entry_price[i] = min(prev_entry, price)
                    position_direction[i] = -1
            
            # Keep existing signal
            signals[i] = signals[i-1]
            continue
        
        # No existing position - look for new entries
        if adx_val < ADX_WEAK:
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_PULLBACK:
                # Pullback entry - determine size based on ADX
                if adx_val > ADX_STRONG:
                    signals[i] = SIZE_FULL
                else:
                    signals[i] = SIZE_HALF
                entry_price[i] = price
                position_direction[i] = 1
            else:
                signals[i] = 0.0
                entry_price[i] = 0
                position_direction[i] = 0
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_PULLBACK:
                # Rally entry - determine size based on ADX
                if adx_val > ADX_STRONG:
                    signals[i] = -SIZE_FULL
                else:
                    signals[i] = -SIZE_HALF
                entry_price[i] = price
                position_direction[i] = -1
            else:
                signals[i] = 0.0
                entry_price[i] = 0
                position_direction[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            entry_price[i] = 0
            position_direction[i] = 0
    
    return signals