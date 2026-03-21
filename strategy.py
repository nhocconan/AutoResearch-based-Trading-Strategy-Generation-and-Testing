#!/usr/bin/env python3
"""
EXPERIMENT #019 - MTF EMA+MACD+RSI+Zscore (15m+1h+4h Clean v1)
==================================================================================================
Hypothesis: Current best uses Supertrend for trend. Let's try EMA crossover (simpler, more responsive)
combined with MACD histogram momentum + RSI pullback entries.

Key changes from current best:
- Trend: 4H EMA(21/55) crossover instead of Supertrend (smoother, fewer whipsaws)
- Momentum: 1H MACD histogram confirmation (adds momentum filter)
- Entry: 15m RSI pullback (40-60 range for quality entries)
- Filter: Z-score to avoid extreme entries
- Stoploss: 2.0*ATR trailing stop
- Position size: 0.35 max (discrete levels for fee control)

Why this should work:
- EMA crossover is proven trend filter (used in many winning strategies)
- MACD histogram adds momentum confirmation without complexity
- 15m timeframe has more opportunities than 1h/4h
- Simpler logic = more trades, less overfitting than #018's 7-filter monstrosity
- Based on lessons from #007 (Sharpe=0.078) which used similar MTF structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_macd_rsi_zscore_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD line)
    macd_signal = np.zeros(n)
    multiplier = 2.0 / (signal + 1)
    
    # Find first valid MACD value
    first_valid = slow - 1
    while first_valid < n and macd_line[first_valid] == 0:
        first_valid += 1
    
    if first_valid < n:
        macd_signal[first_valid + signal - 1] = np.mean(macd_line[first_valid:first_valid + signal])
        
        for i in range(first_valid + signal, n):
            macd_signal[i] = (macd_line[i] - macd_signal[i - 1]) * multiplier + macd_signal[i - 1]
    
    histogram = macd_line - macd_signal
    
    return macd_line, macd_signal, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Get 1h data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_1h = get_htf_data(prices, '1h')
        if df_1h is not None and len(df_1h) > 0:
            close_1h = df_1h['close'].values
            high_1h = df_1h['high'].values
            low_1h = df_1h['low'].values
            
            # 1h indicators for momentum
            macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
            
            # Align 1h MACD histogram to 15m timeframe
            macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
        else:
            macd_hist_1h_aligned = np.zeros(n)
    except Exception:
        macd_hist_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        if df_4h is not None and len(df_4h) > 0:
            close_4h = df_4h['close'].values
            
            # 4h EMA crossover for trend
            ema21_4h = calculate_ema(close_4h, period=21)
            ema55_4h = calculate_ema(close_4h, period=55)
            
            # Determine trend direction (1 = bullish, -1 = bearish, 0 = neutral)
            trend_4h = np.zeros(len(close_4h))
            for i in range(55, len(close_4h)):
                if ema21_4h[i] > ema55_4h[i] and ema21_4h[i-1] <= ema55_4h[i-1]:
                    trend_4h[i] = 1  # Bullish crossover
                elif ema21_4h[i] < ema55_4h[i] and ema21_4h[i-1] >= ema55_4h[i-1]:
                    trend_4h[i] = -1  # Bearish crossover
                elif ema21_4h[i] > ema55_4h[i]:
                    trend_4h[i] = 1  # Already bullish
                elif ema21_4h[i] < ema55_4h[i]:
                    trend_4h[i] = -1  # Already bearish
            
            # Align 4h trend to 15m timeframe
            trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
        else:
            trend_4h_aligned = np.zeros(n)
    except Exception:
        trend_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # MACD histogram threshold for momentum
    MACD_MIN = 0.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 55 * 4, 26 + 9, 20, 14 * 2)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_aligned[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        macd_hist_1h = macd_hist_1h_aligned[i]
        macd_hist_15m = macd_hist_15m[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4H EMA trend + 1H MACD momentum + 15m RSI pullback + Z-score filter
        if trend == 1:  # Bullish trend on 4H
            if (macd_hist_1h > MACD_MIN and  # 1H momentum positive
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and  # 15m RSI pullback
                abs(zscore_val) < ZSCORE_MAX):  # Not extreme
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1:  # Bearish trend on 4H
            if (macd_hist_1h < -MACD_MIN and  # 1H momentum negative
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and  # 15m RSI pullback
                abs(zscore_val) < ZSCORE_MAX):  # Not extreme
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals