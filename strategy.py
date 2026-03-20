#!/usr/bin/env python3
"""
EXPERIMENT #005 - KAMA Adaptive Trend + MACD Momentum + ADX Strength Filter
============================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratios,
providing better trend signals than fixed MAs during varying volatility regimes.
Combined with MACD histogram crosses for momentum entries and ADX strength filter,
this should reduce whipsaw during choppy periods while capturing strong trends.

Key differences from mtf_donchian_rsi_atr_v1:
- KAMA(4h) adaptive trend instead of Donchian breakout (responds to volatility)
- MACD histogram cross on 1h for entry timing (momentum-based vs RSI pullback)
- ADX(14) > 25 filter to ensure we only trade strong trends
- Simpler ATR trailing stop with proper entry price tracking

Why this might beat Sharpe=2.931:
- KAMA reduces lag during trends, increases smoothing during chop
- MACD histogram crosses capture momentum shifts earlier than RSI levels
- ADX filter avoids trading during weak/consolidation periods
- Multi-timeframe: 4h KAMA trend + 1h MACD entries + ADX regime filter
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA calculation helper
    def ema(data, period):
        result = np.zeros(n)
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize sums
    plus_sum = np.sum(plus_dm[1:period + 1])
    minus_sum = np.sum(minus_dm[1:period + 1])
    tr_sum = np.sum(tr[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100 * plus_sum / tr_sum if tr_sum > 0 else 0
            minus_di[i] = 100 * minus_sum / tr_sum if tr_sum > 0 else 0
        else:
            plus_sum = plus_sum - plus_dm[i - 1] + plus_dm[i]
            minus_sum = minus_sum - minus_dm[i - 1] + minus_dm[i]
            tr_sum = tr_sum - tr[i - 1] + tr[i]
            
            plus_di[i] = 100 * plus_sum / tr_sum if tr_sum > 0 else 0
            minus_di[i] = 100 * minus_sum / tr_sum if tr_sum > 0 else 0
        
        # Calculate DX
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Calculate ADX (smoothed DX)
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    macd_line_1h, signal_line_1h, histogram_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h KAMA for adaptive trend (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(20, n_4h):
        if kama_4h[i] > 0 and kama_4h[i - 1] > 0:
            kama_slope = kama_4h[i] - kama_4h[i - 1]
            price_vs_kama = c_4h[i] - kama_4h[i]
            
            if kama_slope > 0 and price_vs_kama > 0:
                trend_4h[i] = 1  # Bullish
            elif kama_slope < 0 and price_vs_kama < 0:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], n_4h - 1)
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in strong trend
    SIZE_HALF = 0.20   # Reduced position in moderate trend
    
    # MACD histogram thresholds for momentum entries
    MACD_LONG_THRESHOLD = 0.0    # Histogram crosses above zero
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crosses below zero
    
    # ADX threshold for trend strength
    ADX_MIN = 25.0  # Only trade when ADX indicates strong trend
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 30, 28)  # Wait for all indicators (KAMA, ADX, MACD)
    
    # Track entry prices for trailing stop
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        if np.isnan(histogram_1h[i]) or np.isnan(adx_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        hist = histogram_1h[i]
        hist_prev = histogram_1h[i - 1] if i > 0 else 0
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            # Check if we have existing position - apply trailing stop
            if position_side[i - 1] != 0 and i > 0:
                if position_side[i - 1] == 1:  # Long
                    stoploss_price = entry_price[i - 1] - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_side[i] = position_side[i - 1]
                        entry_price[i] = entry_price[i - 1]
                elif position_side[i - 1] == -1:  # Short
                    stoploss_price = entry_price[i - 1] + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_side[i] = position_side[i - 1]
                        entry_price[i] = entry_price[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions first
        if i > 0 and position_side[i - 1] != 0:
            if position_side[i - 1] == 1:  # Long
                stoploss_price = entry_price[i - 1] - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif position_side[i - 1] == -1:  # Short
                stoploss_price = entry_price[i - 1] + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        if trend == 1:  # 4h uptrend
            # MACD histogram cross above zero for long entry
            if hist > MACD_LONG_THRESHOLD and hist_prev <= MACD_LONG_THRESHOLD:
                signals[i] = SIZE_FULL
                entry_price[i] = price
                position_side[i] = 1
            elif hist > 0 and position_side[i - 1] == 1:
                # Hold existing long position
                signals[i] = signals[i - 1]
                entry_price[i] = entry_price[i - 1]
                position_side[i] = position_side[i - 1]
            elif hist <= 0 and position_side[i - 1] == 1:
                # MACD turned negative - exit long
                signals[i] = 0.0
                entry_price[i] = 0
                position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
        elif trend == -1:  # 4h downtrend
            # MACD histogram cross below zero for short entry
            if hist < MACD_SHORT_THRESHOLD and hist_prev >= MACD_SHORT_THRESHOLD:
                signals[i] = -SIZE_FULL
                entry_price[i] = price
                position_side[i] = -1
            elif hist < 0 and position_side[i - 1] == -1:
                # Hold existing short position
                signals[i] = signals[i - 1]
                entry_price[i] = entry_price[i - 1]
                position_side[i] = position_side[i - 1]
            elif hist >= 0 and position_side[i - 1] == -1:
                # MACD turned positive - exit short
                signals[i] = 0.0
                entry_price[i] = 0
                position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals