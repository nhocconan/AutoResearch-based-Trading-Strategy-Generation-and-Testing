#!/usr/bin/env python3
"""
EXPERIMENT #012 - KAMA Trend + MACD Histogram + Z-score Filter
====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than EMA/HMA - it speeds up during trends and slows during chop. Combined with MACD 
histogram crosses for precise entry timing (different from RSI pullbacks) and Z-score 
filter (proven in #005 best performer) for regime detection.

Key differences from current best (#005 EMA+RSI+Z-score):
- KAMA(10,2,30) trend instead of EMA - adapts to volatility regimes automatically
- MACD histogram crosses instead of RSI pullbacks - different entry trigger
- Z-score filter (same as #005 proven winner) - avoids extreme mean-reversion traps
- 4h KAMA trend + 1h MACD entries (proven MTF structure)
- Trailing stoploss at 2*ATR, take profit at 2R (reduce to half)
- Discrete signal levels: 0.0, ±0.25, ±0.35 to minimize churn costs

Why this might beat Sharpe=5.525:
- KAMA's adaptive nature handles crypto's varying volatility better than fixed EMA
- MACD histogram crosses provide earlier entry signals than RSI pullbacks
- Z-score filter proven effective in #005 (best performer)
- Multi-timeframe structure proven to 2x Sharpe vs single timeframe
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - fast during trends, slow during chop
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line (EMA of MACD)
    signal_line[slow + signal - 2] = np.mean(macd_line[slow - 1:slow + signal - 1])
    
    for i in range(slow + signal - 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    # Histogram
    for i in range(slow + signal - 2, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


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
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_1h = calculate_zscore(close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA for trend
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(35, n_4h):  # Need enough data for KAMA
        if kama_4h[i] > 0:
            kama_slope = kama_4h[i] - kama_4h[i - 3]  # 3-bar slope (12h)
            if c_4h[i] > kama_4h[i] and kama_slope > 0:
                trend_4h[i] = 1  # Bullish
            elif c_4h[i] < kama_4h[i] and kama_slope < 0:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # MACD histogram thresholds for entry
    MACD_LONG_THRESHOLD = 0  # Histogram crosses above 0
    MACD_SHORT_THRESHOLD = 0  # Histogram crosses below 0
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.0   # Avoid extreme overbought (> 2 std)
    ZSCORE_MIN = -2.0  # Avoid extreme oversold (< -2 std)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(100, 35, 40, 28)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    initial_stop = np.zeros(n)  # Track initial stoploss level
    
    for i in range(first_valid, n):
        if np.isnan(hist_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        zscore_val = zscore_1h[i]
        hist_val = hist_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score regime filter - avoid extreme moves (mean reversion likely)
        if zscore_val > ZSCORE_MAX or zscore_val < ZSCORE_MIN:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price) if position_side[i-1] == 1 else highest_since_entry[i-1]
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price) if position_side[i-1] == -1 else lowest_since_entry[i-1]
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_stop = initial_stop[i - 1] if initial_stop[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Trailing stoploss (2*ATR from entry, or trail from highest)
                trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr if highest_since_entry[i] > 0 else prev_entry - ATR_STOP_MULT * atr
                stoploss_price = max(prev_entry - ATR_STOP_MULT * atr, trail_stop)
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Trailing stoploss
                trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr if lowest_since_entry[i] > 0 else prev_entry + ATR_STOP_MULT * atr
                stoploss_price = min(prev_entry + ATR_STOP_MULT * atr, trail_stop)
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
        
        # Entry logic with MACD histogram confirmation
        position_size = SIZE_FULL
        
        if trend == 1:  # 4h uptrend + Z-score OK
            # MACD histogram crosses above 0 (bullish momentum)
            if hist_val > MACD_LONG_THRESHOLD:
                # Check previous bar for histogram crossing up
                if i > 0 and hist_1h[i-1] <= 0 and hist_1h[i] > 0:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price - ATR_STOP_MULT * atr
                else:
                    # Hold existing position
                    if i > 0 and position_side[i - 1] == 1:
                        signals[i] = signals[i - 1]
                        position_side[i] = 1
                        entry_price[i] = entry_price[i - 1]
                        tp_triggered[i] = tp_triggered[i - 1]
                        highest_since_entry[i] = highest_since_entry[i-1]
                        lowest_since_entry[i] = lowest_since_entry[i-1]
                        initial_stop[i] = initial_stop[i-1]
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend + Z-score OK
            # MACD histogram crosses below 0 (bearish momentum)
            if hist_val < MACD_SHORT_THRESHOLD:
                # Check previous bar for histogram crossing down
                if i > 0 and hist_1h[i-1] >= 0 and hist_1h[i] < 0:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price + ATR_STOP_MULT * atr
                else:
                    # Hold existing position
                    if i > 0 and position_side[i - 1] == -1:
                        signals[i] = signals[i - 1]
                        position_side[i] = -1
                        entry_price[i] = entry_price[i - 1]
                        tp_triggered[i] = tp_triggered[i - 1]
                        highest_since_entry[i] = highest_since_entry[i-1]
                        lowest_since_entry[i] = lowest_since_entry[i-1]
                        initial_stop[i] = initial_stop[i-1]
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            initial_stop[i] = 0
    
    return signals