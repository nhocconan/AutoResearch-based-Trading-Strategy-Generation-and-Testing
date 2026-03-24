#!/usr/bin/env python3
import numpy as np
import pandas as pd

# Module-level metadata (repo contract)
name = "MACD+RSI Momentum"
timeframe = "1h"
leverage = 1

# Strategy Parameters (defaults from Pine Script)
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
SRC_TYPE = "close"

# MA Types
SMA_SOURCE1 = "EMA"
SMA_SOURCE2 = "EMA"
SMA_SIGNAL = "EMA"

# Stress
USE_STRESS = False
RECENT_STRESS = 0.01
STRESS_LEVEL = 1

# Additional MA on MACD
USE_MA = True
MA_LENGTH = 36
MA_TYPE = "THMA"

# Linear Regression
USE_LINREG = True
LINREG_LENGTH = 10
LINREG_OFFSET = 1

# Conditions
ON_CROSS = False
ON_MINMAX = True

# RSI
RSI_FILTER = False
RSI_TP = True
RSI_LEN = 14
RSI_OB = 90
RSI_OS = 44

# Direction
USE_LONG = True
USE_SHORT = True


def _calculate_ma(series, length, ma_type):
    """Calculate moving average with support for multiple MA types."""
    if length < 1:
        return series.copy()
    
    series = pd.Series(series)
    
    if ma_type == "SMA":
        return series.rolling(window=length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "RMA":
        return series.ewm(alpha=1/length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, length + 1)
        def wma(x):
            if np.any(np.isnan(x)):
                return np.nan
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=length).apply(wma, raw=True)
    elif ma_type == "HMA":
        half = max(1, int(length / 2))
        sqrt_len = max(1, int(np.sqrt(length)))
        wma_full = _calculate_ma(series, length, "WMA")
        wma_half = _calculate_ma(series, half, "WMA")
        raw_hma = 2 * wma_half - wma_full
        return _calculate_ma(raw_hma, sqrt_len, "WMA")
    elif ma_type.startswith("D"):
        base = ma_type[1:]
        ma1 = _calculate_ma(series, length, base)
        ma2 = _calculate_ma(ma1, length, base)
        return 2 * ma1 - ma2
    elif ma_type.startswith("T"):
        base = ma_type[1:]
        ma1 = _calculate_ma(series, length, base)
        ma2 = _calculate_ma(ma1, length, base)
        ma3 = _calculate_ma(ma2, length, base)
        return 3 * ma1 - ma3
    elif ma_type.startswith("F"):
        base = ma_type[1:]
        ma1 = _calculate_ma(series, length, base)
        ma2 = _calculate_ma(ma1, length, base)
        ma3 = _calculate_ma(ma2, length, base)
        ma4 = _calculate_ma(ma3, length, base)
        return 4 * ma1 - ma4
    else:
        return series.ewm(span=length, adjust=False).mean()


def _calculate_rsi(series, length):
    """Calculate RSI using RMA logic as per TradingView standard."""
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_linreg(series, length, offset):
    """Calculate linear regression with offset."""
    series = pd.Series(series)
    n = len(series)
    res = np.full(n, np.nan)
    
    for i in range(length - 1, n):
        window = series.iloc[i - length + 1 : i + 1].values
        if np.any(np.isnan(window)):
            continue
        x = np.arange(length)
        coeffs = np.polyfit(x, window, 1)
        target_idx = length - 1 - offset
        if 0 <= target_idx < length:
            res[i] = coeffs[0] * target_idx + coeffs[1]
    
    return pd.Series(res, index=series.index)


def generate_signals(prices):
    """
    Generate trading signals based on MACD+RSI strategy.
    
    Args:
        prices: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        numpy array with exactly len(prices) elements containing signals:
        1 = Long, -1 = Short, 0 = Flat
    """
    df = prices.copy()
    n = len(df)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Initialize signal array
    signals = np.zeros(n, dtype=np.float64)
    
    # Get source data
    src = df['close']
    
    # 1. Base MACD
    fast_ma = _calculate_ma(src, FAST_LENGTH, SMA_SOURCE1)
    slow_ma = _calculate_ma(src, SLOW_LENGTH, SMA_SOURCE2)
    macd = fast_ma - slow_ma
    
    # 2. Signal Line
    signal_line = _calculate_ma(macd, SIGNAL_LENGTH, SMA_SIGNAL)
    
    # 3. Stress Logic (Iterative)
    if USE_STRESS:
        stressed_macd = np.zeros(n)
        for i in range(n):
            if np.isnan(macd.iloc[i]):
                stressed_macd[i] = np.nan
                continue
            val = macd.iloc[i] * (1.0 / (1.0 - RECENT_STRESS))
            if i > 0 and not np.isnan(stressed_macd[i-1]):
                prev = stressed_macd[i-1]
                val = pow((val * RECENT_STRESS), STRESS_LEVEL) + (1.0 - RECENT_STRESS * prev)
            stressed_macd[i] = val
        macd = pd.Series(stressed_macd, index=df.index)
    
    # 4. Additional MA on MACD
    if USE_MA:
        macd = _calculate_ma(macd, MA_LENGTH, MA_TYPE)
    
    # 5. Linear Regression on MACD
    if USE_LINREG:
        macd = _calculate_linreg(macd, LINREG_LENGTH, LINREG_OFFSET)
    
    # 6. RSI
    rsi = _calculate_rsi(src, RSI_LEN)
    
    # 7. Conditions
    change_macd = macd.diff()
    change_macd_prev = change_macd.shift(1)
    
    # Entry Conditions Base
    if ON_CROSS:
        apertura_long = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        apertura_short = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
    else:
        # on_minmax default
        apertura_long = change_macd > 0
        apertura_short = change_macd < 0
    
    # OK Conditions (RSI Filter)
    if RSI_FILTER:
        ok_long = (rsi < RSI_OB) & (change_macd > 0) & (change_macd_prev <= 0)
        ok_short = (rsi > RSI_OS) & (change_macd < 0) & (change_macd_prev >= 0)
    else:
        ok_long = pd.Series(True, index=df.index)
        ok_short = pd.Series(True, index=df.index)
    
    # RSI TP Logic
    if RSI_TP:
        apertura_long = (change_macd > 0) & (change_macd_prev <= 0) & (rsi < RSI_OB)
        apertura_short = (change_macd < 0) & (change_macd_prev >= 0) & (rsi > RSI_OS)
        
        chiusura_short = (rsi < RSI_OS) | (apertura_long)
        chiusura_long = (rsi > RSI_OB) | (apertura_short)
    else:
        chiusura_short = ok_long & apertura_long
        chiusura_long = ok_short & apertura_short
    
    # Fill boolean series with False for NaN values
    apertura_long = apertura_long.fillna(False)
    apertura_short = apertura_short.fillna(False)
    ok_long = ok_long.fillna(False)
    ok_short = ok_short.fillna(False)
    chiusura_short = chiusura_short.fillna(False)
    chiusura_long = chiusura_long.fillna(False)
    
    # 8. Simulation Loop for Position (no lookahead, next-bar execution)
    position = 0  # 0: Flat, 1: Long, -1: Short
    
    for i in range(n):
        if i == 0:
            signals[i] = 0
            continue
        
        # Check Close Conditions first (based on previous bar signals)
        if position == -1:  # Short
            if chiusura_short.iloc[i]:
                position = 0
        elif position == 1:  # Long
            if chiusura_long.iloc[i]:
                position = 0
        
        # Check Entry Conditions (signals execute on next bar)
        if position == 0:
            if USE_LONG and ok_long.iloc[i] and apertura_long.iloc[i]:
                position = 1
            elif USE_SHORT and ok_short.iloc[i] and apertura_short.iloc[i]:
                position = -1
        
        signals[i] = position
    
    return signals


if __name__ == "__main__":
    pass
