#!/usr/bin/env python3
"""
EXPERIMENT #012 - Supertrend Trend + MACD Histogram Entry + ATR Position Sizing
===============================================================================
Hypothesis: Supertrend provides cleaner trend signals than moving averages with built-in
volatility adjustment. Combined with MACD histogram for momentum entry timing and
ATR-based dynamic position sizing, this should reduce drawdown while capturing trends.

Key differences from mtf_hma_supertrend_rsi_v1:
- Supertrend(10,3) for trend instead of HMA (volatility-adjusted trend)
- MACD histogram cross for entry timing instead of RSI pullback
- ATR-based position sizing (smaller positions when volatility is high)
- Multi-timeframe: 4h Supertrend trend + 1h MACD entries

Why this might beat Sharpe=2.931:
- Supertrend adapts to volatility regime automatically
- MACD histogram captures momentum shifts earlier than RSI
- Dynamic position sizing reduces risk in high-vol periods
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_atr_v1"
timeframe = "1h"
leverage = 1.0


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            # Update supertrend based on previous trend
            if trend[i - 1] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    trend[i] = -1
    
    return supertrend, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    # Initialize with SMA for first values
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    # Calculate EMAs
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] * (fast - 1) / (fast + 1) + close[i] * 2 / (fast + 1)
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] * (slow - 1) / (slow + 1) + close[i] * 2 / (slow + 1)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    first_signal = slow + signal - 1
    if first_signal < n:
        signal_line[first_signal] = np.mean(macd_line[slow:first_signal + 1])
        for i in range(first_signal + 1, n):
            signal_line[i] = signal_line[i - 1] * (signal - 1) / (signal + 1) + macd_line[i] * 2 / (signal + 1)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    rsi_1h = calculate_rsi(close, period=14)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    _, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    BASE_SIZE = 0.30   # Base position size
    SIZE_MIN = 0.15    # Minimum position in high vol
    SIZE_MAX = 0.35    # Maximum position in low vol
    
    # MACD histogram thresholds for entry
    MACD_LONG_THRESHOLD = 0.0    # Histogram crosses above 0
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crosses below 0
    
    # ATR volatility filter
    ATR_VOL_TARGET = 0.02  # Target ATR as % of price (2%)
    ATR_VOL_MAX = 0.05     # Max ATR as % of price (5%)
    
    # RSI filter for overbought/oversold
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    first_valid = max(80, 26, 35, 14)  # Wait for all indicators
    
    # Track position state for stoploss
    position_state = np.zeros(n)  # 0=none, 1=long, -1=short
    entry_price = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_histogram = macd_hist[i]
        rsi_val = rsi_1h[i]
        
        # ATR volatility filter - avoid trading when too volatile
        atr_pct = atr / price if price > 0 else 0
        if atr_pct > ATR_VOL_MAX:
            signals[i] = 0.0
            position_state[i] = 0
            continue
        
        # Dynamic position sizing based on ATR
        if atr_pct > 0:
            vol_factor = ATR_VOL_TARGET / atr_pct
            vol_factor = np.clip(vol_factor, 0.5, 1.5)  # Limit sizing adjustment
            position_size = BASE_SIZE * vol_factor
            position_size = np.clip(position_size, SIZE_MIN, SIZE_MAX)
        else:
            position_size = BASE_SIZE
        
        # Check existing position for stoploss
        if position_state[i - 1] != 0 and i > 0:
            prev_entry = entry_price[i - 1]
            prev_state = position_state[i - 1]
            stoploss_distance = 2.5 * atr  # 2.5 ATR stoploss
            
            if prev_state == 1:  # Long position
                stoploss_price = prev_entry - stoploss_distance
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_state[i] = 0
                    continue
            elif prev_state == -1:  # Short position
                stoploss_price = prev_entry + stoploss_distance
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_state[i] = 0
                    continue
        
        # Entry logic based on trend and MACD
        if trend == 1:  # 4h uptrend - look for long entries
            # RSI filter - don't buy at overbought
            if rsi_val > RSI_OVERBOUGHT:
                signals[i] = 0.0
                continue
            
            # MACD histogram confirmation
            if macd_histogram > MACD_LONG_THRESHOLD:
                # Check for histogram turning up (momentum confirmation)
                if i > 0 and macd_hist[i - 1] <= macd_histogram:
                    signals[i] = position_size
                    position_state[i] = 1
                    entry_price[i] = price
                else:
                    # Hold existing position
                    if position_state[i - 1] == 1:
                        signals[i] = signals[i - 1]
                        position_state[i] = 1
                        entry_price[i] = entry_price[i - 1]
                    else:
                        signals[i] = 0.0
                        position_state[i] = 0
            else:
                # Hold or exit
                if position_state[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_state[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_state[i] = 0
                    
        elif trend == -1:  # 4h downtrend - look for short entries
            # RSI filter - don't sell at oversold
            if rsi_val < RSI_OVERSOLD:
                signals[i] = 0.0
                continue
            
            # MACD histogram confirmation
            if macd_histogram < MACD_SHORT_THRESHOLD:
                # Check for histogram turning down (momentum confirmation)
                if i > 0 and macd_hist[i - 1] >= macd_histogram:
                    signals[i] = -position_size
                    position_state[i] = -1
                    entry_price[i] = price
                else:
                    # Hold existing position
                    if position_state[i - 1] == -1:
                        signals[i] = signals[i - 1]
                        position_state[i] = -1
                        entry_price[i] = entry_price[i - 1]
                    else:
                        signals[i] = 0.0
                        position_state[i] = 0
            else:
                # Hold or exit
                if position_state[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_state[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_state[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_state[i] = 0
    
    # Smooth signals to reduce churn (only change at significant levels)
    for i in range(1, n):
        if signals[i] != 0 and signals[i - 1] != 0:
            # Same direction - keep
            if np.sign(signals[i]) == np.sign(signals[i - 1]):
                signals[i] = signals[i - 1]  # Keep previous size to reduce churn
    
    return signals