#!/usr/bin/env python3
"""MACD+RSI Momentum Strategy - Converted from TradingView Pine Script"""

import numpy as np
import pandas as pd

name = "MACD+RSI Momentum Strategy (BTC/USDT; 1h)"
timeframe = "1h"
leverage = 1

# Default parameters from Pine Script
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
RSI_LENGTH = 14
RSI_OVERBOUGHT = 90
RSI_OVERSOLD = 44
USE_STRESS = False
USE_MA_SMOOTH = True
MA_SMOOTH_LENGTH = 36
USE_LINREG = True
LINREG_LENGTH = 10
ON_CROSS = False
ON_MINMAX = True
USE_RSI_FILTER = False
USE_RSI_TP = True
USE_LONG = True
USE_SHORT = True


def _ema(series, length):
    """Calculate Exponential Moving Average"""
    if length <= 0:
        return series.copy()
    alpha = 2.0 / (length + 1)
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _sma(series, length):
    """Calculate Simple Moving Average"""
    if length <= 0:
        return series.copy()
    result = np.zeros_like(series)
    for i in range(len(series)):
        if i < length - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(series[i - length + 1:i + 1])
    return result


def _wma(series, length):
    """Calculate Weighted Moving Average"""
    if length <= 0:
        return series.copy()
    result = np.zeros_like(series)
    weights = np.arange(1, length + 1)
    for i in range(len(series)):
        if i < length - 1:
            result[i] = np.nan
        else:
            window = series[i - length + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
    return result


def _rma(series, length):
    """Calculate Running Moving Average (Wilder's smoothing)"""
    if length <= 0:
        return series.copy()
    result = np.zeros_like(series)
    alpha = 1.0 / length
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _hma(series, length):
    """Calculate Hull Moving Average"""
    if length <= 0:
        return series.copy()
    half = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma_half = _wma(series, half)
    wma_full = _wma(series, length)
    diff = 2 * wma_half - wma_full
    return _wma(diff, sqrt_len)


def _ma(series, length, ma_type):
    """Calculate moving average based on type"""
    if ma_type == "EMA":
        return _ema(series, length)
    elif ma_type == "SMA":
        return _sma(series, length)
    elif ma_type == "WMA":
        return _wma(series, length)
    elif ma_type == "RMA":
        return _rma(series, length)
    elif ma_type == "HMA":
        return _hma(series, length)
    elif ma_type == "DEMA":
        ema1 = _ema(series, length)
        ema2 = _ema(ema1, length)
        return 2 * ema1 - ema2
    elif ma_type == "TEMA":
        ema1 = _ema(series, length)
        ema2 = _ema(ema1, length)
        ema3 = _ema(ema2, length)
        return 3 * ema1 - 3 * ema2 + ema3
    elif ma_type == "THMA":
        hma1 = _hma(series, length)
        hma2 = _hma(hma1, length)
        hma3 = _hma(hma2, length)
        return 3 * hma1 - 3 * hma2 + hma3
    else:
        return _ema(series, length)


def _rsi(series, length):
    """Calculate Relative Strength Index"""
    if length <= 0:
        return np.zeros_like(series)
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    avg_gain[0] = np.nan
    avg_loss[0] = np.nan
    for i in range(1, len(series)):
        if i < length:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == length:
            avg_gain[i] = np.mean(gain[:length])
            avg_loss[i] = np.mean(loss[:length])
        else:
            avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain[i - 1]) / length
            avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss[i - 1]) / length
    rs = np.zeros_like(series)
    for i in range(len(series)):
        if avg_loss[i] == 0 or np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            rs[i] = np.nan
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    rsi_values = np.zeros_like(series)
    for i in range(len(series)):
        if np.isnan(rs[i]):
            rsi_values[i] = np.nan
        else:
            rsi_values[i] = 100 - (100 / (1 + rs[i]))
    return rsi_values


def _linreg(series, length, offset=0):
    """Calculate Linear Regression"""
    if length <= 0:
        return series.copy()
    result = np.zeros_like(series)
    for i in range(len(series)):
        if i < length - 1:
            result[i] = np.nan
        else:
            y = series[i - length + 1:i + 1]
            x = np.arange(length)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            pred_idx = length - 1 + offset
            if pred_idx < 0 or pred_idx >= length:
                result[i] = y[-1]
            else:
                result[i] = intercept + slope * pred_idx
    return result


def _apply_stress(macd, use_stress, recent_stress, level):
    """Apply stress adjustment to MACD"""
    if not use_stress:
        return macd.copy()
    result = macd.copy()
    for i in range(len(macd)):
        if np.isnan(macd[i]):
            continue
        result[i] = macd[i] * (1 / (1 - recent_stress))
        if i > 0 and not np.isnan(macd[i - 1]):
            result[i] = np.power((macd[i] * recent_stress), level) + (1 - recent_stress * macd[i - 1])
    return result


def generate_signals(prices):
    """Generate target position signals based on MACD+RSI strategy"""
    n = len(prices)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    close = prices['close'].values.astype(np.float64)
    
    # Calculate MACD
    fast_ma = _ma(close, FAST_LENGTH, "EMA")
    slow_ma = _ma(close, SLOW_LENGTH, "EMA")
    macd = fast_ma - slow_ma
    signal_line = _ma(macd, SIGNAL_LENGTH, "EMA")
    
    # Apply optional stress
    macd = _apply_stress(macd, USE_STRESS, 0.01, 1)
    
    # Apply optional MA smoothing
    if USE_MA_SMOOTH:
        macd = _ma(macd, MA_SMOOTH_LENGTH, "THMA")
    
    # Apply optional linear regression
    if USE_LINREG:
        macd = _linreg(macd, LINREG_LENGTH, 1)
    
    # Calculate RSI
    rsi = _rsi(close, RSI_LENGTH)
    
    # Calculate change in MACD
    macd_change = np.zeros_like(macd)
    macd_change[0] = np.nan
    for i in range(1, n):
        if np.isnan(macd[i]) or np.isnan(macd[i - 1]):
            macd_change[i] = np.nan
        else:
            macd_change[i] = macd[i] - macd[i - 1]
    
    # Entry conditions
    if ON_CROSS:
        apertura_long = np.zeros(n, dtype=bool)
        apertura_short = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(macd[i]) and not np.isnan(macd[i - 1]) and \
               not np.isnan(signal_line[i]) and not np.isnan(signal_line[i - 1]):
                if macd[i - 1] <= signal_line[i - 1] and macd[i] > signal_line[i]:
                    apertura_long[i] = True
                if macd[i - 1] >= signal_line[i - 1] and macd[i] < signal_line[i]:
                    apertura_short[i] = True
    else:
        apertura_long = np.zeros(n, dtype=bool)
        apertura_short = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(macd_change[i]) and not np.isnan(macd_change[i - 1]):
                if ON_MINMAX:
                    if macd_change[i] > 0 and macd_change[i - 1] <= 0:
                        apertura_long[i] = True
                    if macd_change[i] < 0 and macd_change[i - 1] >= 0:
                        apertura_short[i] = True
                else:
                    if macd_change[i] > 0:
                        apertura_long[i] = True
                    if macd_change[i] < 0:
                        apertura_short[i] = True
    
    # RSI filter conditions
    ok_long = np.ones(n, dtype=bool)
    ok_short = np.ones(n, dtype=bool)
    
    if USE_RSI_FILTER or USE_RSI_TP:
        for i in range(n):
            if not np.isnan(rsi[i]):
                if USE_RSI_TP:
                    if USE_RSI_FILTER:
                        ok_long[i] = (rsi[i] < RSI_OVERBOUGHT) and (macd_change[i] > 0 if i > 0 else False)
                        if i > 0 and not np.isnan(macd_change[i - 1]):
                            ok_long[i] = ok_long[i] and (macd_change[i - 1] <= 0)
                        ok_short[i] = (rsi[i] > RSI_OVERSOLD) and (macd_change[i] < 0 if i > 0 else False)
                        if i > 0 and not np.isnan(macd_change[i - 1]):
                            ok_short[i] = ok_short[i] and (macd_change[i - 1] >= 0)
                else:
                    ok_long[i] = (rsi[i] < RSI_OVERBOUGHT)
                    ok_short[i] = (rsi[i] > RSI_OVERSOLD)
    
    # Exit conditions
    chiusura_long = np.zeros(n, dtype=bool)
    chiusura_short = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not np.isnan(rsi[i]):
            if USE_RSI_TP:
                chiusura_short[i] = (rsi[i] < RSI_OVERSOLD) or apertura_long[i]
                chiusura_long[i] = (rsi[i] > RSI_OVERBOUGHT) or apertura_short[i]
            else:
                chiusura_short[i] = ok_long[i] and apertura_long[i]
                chiusura_long[i] = ok_short[i] and apertura_short[i]
    
    # Generate position signals (target position)
    signals = np.zeros(n, dtype=np.float64)
    position = 0  # 0 = flat, 1 = long, -1 = short
    
    for i in range(n):
        if position == 0:
            if USE_LONG and ok_long[i] and apertura_long[i]:
                position = 1
                signals[i] = 1.0
            elif USE_SHORT and ok_short[i] and apertura_short[i]:
                position = -1
                signals[i] = -1.0
        elif position == 1:
            if chiusura_long[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 1.0
        elif position == -1:
            if chiusura_short[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -1.0
    
    return signals
