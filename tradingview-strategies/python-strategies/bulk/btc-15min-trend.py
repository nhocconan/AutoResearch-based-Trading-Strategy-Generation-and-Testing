#!/usr/bin/env python3
"""
BTC 15 min Strategy - Converted from TradingView Pine Script
Approximates MTF logic using available 15m data only.
"""

import numpy as np
import pandas as pd

name = "BTC 15 min"
timeframe = "15m"
leverage = 1


def sma(series, length):
    return series.rolling(window=length, min_periods=length).mean()


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def wma(series, length):
    weights = np.arange(1, length + 1, dtype=float)
    def apply_wma(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window=length, min_periods=length).apply(apply_wma, raw=True)


def stdev(series, length):
    return series.rolling(window=length, min_periods=length).std()


def linreg(series, length, offset):
    x = np.arange(length)
    def calc_linreg(window):
        if len(window) < length or np.any(np.isnan(window)):
            return np.nan
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (length - 1 + offset)
    return series.rolling(window=length, min_periods=length).apply(calc_linreg, raw=True)


def rsi(series, length):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = ema(gain, length)
    avg_loss = ema(loss, length)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def generate_signals(prices):
    """
    Generate trading signals from price data.
    
    Args:
        prices: pandas DataFrame with columns:
                open_time, open, high, low, close, volume
    
    Returns:
        numpy array of signals: 1=long, -1=short, 0=neutral
    """
    if isinstance(prices, np.ndarray):
        df = pd.DataFrame({
            'open_time': np.arange(len(prices)),
            'open': prices[:, 0] if prices.ndim > 1 else prices,
            'high': prices[:, 1] if prices.ndim > 1 else prices,
            'low': prices[:, 2] if prices.ndim > 1 else prices,
            'close': prices[:, 3] if prices.ndim > 1 else prices,
            'volume': prices[:, 4] if prices.ndim > 1 else np.ones(len(prices))
        })
    else:
        df = prices.copy()
    
    n = len(df)
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    
    close_series = pd.Series(close)
    
    length8 = 30
    upmult = 5.0
    lowmult = 5.0
    
    basis = sma(close_series, length8).values
    vup = upmult * close / 100
    vlow = lowmult * close / 100
    upper = basis + vup
    lower = basis - vlow
    
    fastLength = 3
    slowLength = 21
    v1 = ema(close_series, fastLength).values
    v2 = ema(close_series, slowLength).values
    
    rsi_period = 14
    vrsi = rsi(close_series, rsi_period).values
    len55 = 10
    pp = wma(pd.Series(vrsi), len55).values
    
    d = np.roll(vrsi, 1) - np.roll(pp, 1)
    d[0] = 0
    len100 = 10
    x = ema(pd.Series(d), len100).values
    zx = x / -1
    
    length_hma = 50
    shift = 1
    p = length_hma / 2
    wma_p3 = wma(close_series, max(1, int(p/3))).values
    wma_p2 = wma(close_series, max(1, int(p/2))).values
    wma_p = wma(close_series, max(1, int(p))).values
    b_raw = wma_p3 * 3 - wma_p2 - wma_p
    b = wma(pd.Series(b_raw), max(1, int(p))).values
    b = np.roll(b, shift)
    b[:shift] = np.nan
    
    len_reg = 25
    linear_reg = linreg(close_series, len_reg, 0).values
    
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(linear_reg[i]) and not np.isnan(b[i]):
            if not np.isnan(linear_reg[i-1]) and not np.isnan(b[i-1]):
                if linear_reg[i] > b[i] and linear_reg[i-1] <= b[i-1]:
                    buy[i] = True
        if not np.isnan(linear_reg[i]) and not np.isnan(b[i]):
            if not np.isnan(linear_reg[i-1]) and not np.isnan(b[i-1]):
                if linear_reg[i] < b[i] and linear_reg[i-1] >= b[i-1]:
                    sell[i] = True
        if i >= 1 and not np.isnan(upper[i-1]):
            if close[i-1] > upper[i-1] and close[i] <= upper[i]:
                sell[i] = True
    
    Min = 15
    leni = max(1, int(Min / 1 * 7))
    l1 = wma(pd.Series(low), leni).values
    h1 = wma(pd.Series(high), leni).values
    m = (h1 + l1) / 2
    
    len5 = 100
    src5 = m
    multi = 2
    
    mean = ema(pd.Series(src5), len5).values
    stddev = multi * stdev(pd.Series(src5), len5).values
    b5 = mean + stddev
    s5 = mean - stddev
    
    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(src5[i]) and not np.isnan(s5[i]):
            if not np.isnan(src5[i-1]) and not np.isnan(s5[i-1]):
                if src5[i] > s5[i] and src5[i-1] <= s5[i-1]:
                    long_sig[i] = True
        if not np.isnan(src5[i]) and not np.isnan(b5[i]):
            if not np.isnan(src5[i-1]) and not np.isnan(b5[i-1]):
                if src5[i] < b5[i] and src5[i-1] >= b5[i-1]:
                    short_sig[i] = True
    
    longCond = np.zeros(n, dtype=bool)
    shortCond = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(zx[i]) and not np.isnan(zx[i-1]):
            if zx[i] > 0 and zx[i-1] <= 0:
                longCond[i] = True
        if buy[i]:
            longCond[i] = True
        if sell[i]:
            shortCond[i] = True
    
    sectionLongs = np.zeros(n, dtype=int)
    sectionShorts = np.zeros(n, dtype=int)
    pyrl = 1
    
    for i in range(1, n):
        sectionLongs[i] = sectionLongs[i-1]
        sectionShorts[i] = sectionShorts[i-1]
        
        if longCond[i]:
            sectionLongs[i] = sectionLongs[i-1] + 1
            sectionShorts[i] = 0
        
        if shortCond[i]:
            sectionLongs[i] = 0
            sectionShorts[i] = sectionShorts[i-1] + 1
    
    longCondition = longCond & (sectionLongs <= pyrl)
    shortCondition = shortCond & (sectionShorts <= pyrl)
    
    last_open_longCondition = np.zeros(n)
    last_open_shortCondition = np.zeros(n)
    last_longCondition = np.zeros(n)
    last_shortCondition = np.zeros(n)
    
    for i in range(n):
        if i > 0:
            last_open_longCondition[i] = last_open_longCondition[i-1]
            last_open_shortCondition[i] = last_open_shortCondition[i-1]
            last_longCondition[i] = last_longCondition[i-1]
            last_shortCondition[i] = last_shortCondition[i-1]
        
        if longCondition[i]:
            last_open_longCondition[i] = close[i]
            last_longCondition[i] = i
        
        if shortCondition[i]:
            last_open_shortCondition[i] = close[i]
            last_shortCondition[i] = i
    
    in_longCondition = last_longCondition > last_shortCondition
    in_shortCondition = last_shortCondition > last_longCondition
    
    tp = 2.0
    sl = 5.0
    
    long_tp = np.zeros(n, dtype=bool)
    long_sl = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if in_longCondition[i] and last_open_longCondition[i] > 0:
            tp_level = (1 + tp / 100) * last_open_longCondition[i]
            sl_level = (1 - sl / 100) * last_open_longCondition[i]
            
            if high[i] > tp_level:
                long_tp[i] = True
            if low[i] < sl_level:
                long_sl[i] = True
    
    filter_flag = True
    l = ((v1 > v2) | (not filter_flag)) & longCondition | long_sl
    s = shortCondition
    
    signals = np.zeros(n, dtype=int)
    
    for i in range(n):
        if l[i]:
            signals[i] = 1
        elif s[i]:
            signals[i] = -1
    
    return signals


if __name__ == "__main__":
    test_data = pd.DataFrame({
        'open_time': np.arange(100),
        'open': np.random.uniform(40000, 45000, 100),
        'high': np.random.uniform(40000, 45000, 100),
        'low': np.random.uniform(40000, 45000, 100),
        'close': np.random.uniform(40000, 45000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    sigs = generate_signals(test_data)
    print(f"Generated {len(sigs)} signals")
    print(f"Long: {np.sum(sigs == 1)}, Short: {np.sum(sigs == -1)}, Neutral: {np.sum(sigs == 0)}")
