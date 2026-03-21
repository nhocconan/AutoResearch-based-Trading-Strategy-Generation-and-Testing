#!/usr/bin/env python3
"""
EXPERIMENT #010 - KAMA 12h with Daily Trend Filter + BBW Regime
================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA,
reducing whipsaws during choppy periods. Combined with daily trend filter and Bollinger
Band Width regime detection, this should capture trends only during high-volatility expansion
phases while avoiding sideways markets.

Key differences from failed strategies:
- KAMA instead of EMA/HMA/Supertrend (adaptive to market efficiency)
- BBW regime filter (only trade when bands expanding = trending)
- 12h primary + 1d HTF (higher TF than most attempts)
- RSI pullback entry within trend direction
- ATR-based stoploss with signal→0 on stop hit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_12h_daily_bbw_rsi_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |close - close[period]| / sum(|close[i] - close[i-1]|)
    er = np.zeros(n)
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i-period]):
            er[i] = 0
            continue
        price_change = abs(close[i] - close[i-period])
        volatility = 0
        for j in range(i-period+1, i+1):
            volatility += abs(close[j] - close[j-1])
        er[i] = price_change / volatility if volatility > 0 else 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (2.0 + 1.0)  # fast = 2/(2+1)
    slow_sc = 2.0 / (2.0 + 30.0)  # slow = 2/(2+30)
    
    # KAMA calculation
    # kama[i] = kama[i-1] + sc^2 * (close[i] - kama[i-1])
    # where sc = er * (fast_sc - slow_sc) + slow_sc
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
            continue
        sc = er[i] * (fast_sc - slow_sc) + slow_sc
        kama[i] = kama[i-1] + sc * sc * (close[i] - kama[i-1])
    
    return kama


def calculate_bbw(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Bollinger Band Width = (Upper - Lower) / Middle
    Higher BW = more volatility/trending, Lower BW = squeeze/consolidation
    """
    n = len(close)
    bbw = np.zeros(n)
    
    # Rolling mean and std
    for i in range(period, n):
        window = close[i-period+1:i+1]
        if np.any(np.isnan(window)):
            bbw[i] = 0
            continue
        middle = np.mean(window)
        std = np.std(window)
        upper = middle + 2.0 * std
        lower = middle - 2.0 * std
        bbw[i] = (upper - lower) / middle if middle > 0 else 0
    
    return bbw


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation with proper min_periods"""
    n = len(close)
    rsi = np.zeros(n)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            gain[i] = delta[i-1]
        else:
            loss[i] = -delta[i-1]
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0 or np.isnan(avg_loss[i]):
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    for i in range(period, n):
        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR calculation with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily KAMA for trend direction
    daily_close = df_1d['close'].values
    daily_kama_fast = calculate_kama(daily_close, period=10)
    daily_kama_slow = calculate_kama(daily_close, period=30)
    
    # Align daily trend to 12h timeframe
    daily_trend = np.zeros(len(daily_close))
    for i in range(30, len(daily_close)):
        if daily_kama_fast[i] > daily_kama_slow[i]:
            daily_trend[i] = 1  # bullish
        elif daily_kama_fast[i] < daily_kama_slow[i]:
            daily_trend[i] = -1  # bearish
        else:
            daily_trend[i] = 0
    
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Calculate 12h indicators
    kama_fast = calculate_kama(close, period=10)
    kama_slow = calculate_kama(close, period=30)
    bbw = calculate_bbw(close, high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Calculate BBW percentile for regime filter
    # Only trade when BBW is above median (expanding volatility)
    bbw_median = np.nanmedian(bbw[50:])  # skip initial warmup
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_LONG = 0.30  # 30% position size
    SIZE_SHORT = -0.30
    
    # Track position for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Minimum bars for all indicators to be valid
    min_bars = max(50, 30)  # KAMA(30) + RSI(14) + ATR(14)
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(bbw[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(daily_trend_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when BBW above median (trending market)
        in_trending_regime = bbw[i] > bbw_median
        
        # Daily trend direction
        daily_trend_dir = daily_trend_aligned[i]
        
        # 12h KAMA crossover signal
        kama_crossover = 0
        if kama_fast[i] > kama_slow[i]:
            kama_crossover = 1
        elif kama_fast[i] < kama_slow[i]:
            kama_crossover = -1
        
        # RSI pullback filter (enter on pullback within trend)
        rsi_pullback_long = rsi[i] < 50 and rsi[i] > 30  # pullback in uptrend
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 70  # pullback in downtrend
        
        # Generate entry signals
        new_signal = 0.0
        
        if in_trending_regime:
            # Long entry: daily bullish + 12h KAMA bullish + RSI pullback
            if daily_trend_dir == 1 and kama_crossover == 1 and rsi_pullback_long:
                new_signal = SIZE_LONG
            
            # Short entry: daily bearish + 12h KAMA bearish + RSI pullback
            elif daily_trend_dir == -1 and kama_crossover == -1 and rsi_pullback_short:
                new_signal = SIZE_SHORT
        
        # Stoploss logic: ATR-based trailing stop
        if position_side == 1:  # Long position
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr[i]
            if close[i] < stop_price:
                new_signal = 0.0  # Stoploss hit
        elif position_side == -1:  # Short position
            lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr[i]
            if close[i] > stop_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        if signals[i-1] == 0 and new_signal != 0:
            # New entry
            position_side = 1 if new_signal > 0 else -1
            entry_price = close[i]
            highest_price = close[i]
            lowest_price = close[i]
        elif signals[i-1] != 0 and new_signal == 0:
            # Exit
            position_side = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals