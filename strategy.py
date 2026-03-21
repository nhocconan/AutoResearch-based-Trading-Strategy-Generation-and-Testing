#!/usr/bin/env python3
"""
EXPERIMENT #011 - 1h Primary with 4h Supertrend Trend + MACD/RSI Entry
========================================================================
Hypothesis: 1h timeframe provides better entry timing than 12h while 4h Supertrend
gives reliable trend direction without the noise of lower timeframes. MACD histogram
divergence combined with RSI extremes provides high-probability entry signals within
the 4h trend direction. Bollinger Band Width regime filter avoids choppy markets.

Key differences from failed strategies:
- Primary=1h (not 12h or 6h) for better entry timing
- HTF=4h Supertrend (proven trend filter, not KAMA/EMA/HMA)
- MACD histogram momentum + RSI pullback (not just RSI alone)
- BBW percentile regime filter (trade only in top 50% volatility)
- ATR trailing stoploss with signal→0 on stop hit
- Discrete position sizing: 0.0, ±0.25, ±0.35
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_supertrend_macd_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Supertrend indicator - returns (supertrend_values, trend_direction)
    trend_direction: +1 = bullish (price above supertrend), -1 = bearish
    """
    n = len(close)
    supertrend = np.zeros(n)
    trend_dir = np.zeros(n)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate supertrend
    supertrend[0] = upper_band[0]
    trend_dir[0] = 1
    
    for i in range(1, n):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            trend_dir[i] = trend_dir[i-1]
            continue
        
        # Initial supertrend
        if trend_dir[i-1] == 1:
            supertrend[i] = min(upper_band[i], supertrend[i-1]) if supertrend[i-1] < close[i-1] else upper_band[i]
        else:
            supertrend[i] = max(lower_band[i], supertrend[i-1]) if supertrend[i-1] > close[i-1] else lower_band[i]
        
        # Determine trend direction
        if close[i] > supertrend[i]:
            trend_dir[i] = 1
        else:
            trend_dir[i] = -1
    
    return supertrend, trend_dir


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    MACD indicator - returns (macd_line, signal_line, histogram)
    """
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation with proper min_periods"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
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
    
    for i in range(period, n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            continue
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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


def calculate_bbw(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Bollinger Band Width = (Upper - Lower) / Middle
    Higher BW = more volatility/trending
    """
    n = len(close)
    bbw = np.zeros(n)
    bbw[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        middle = np.mean(window)
        std = np.std(window)
        if middle > 0:
            bbw[i] = (4.0 * std) / middle  # (upper-lower)/middle = 4*std/middle
    
    return bbw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Supertrend for trend direction
    htf_close = df_4h['close'].values
    htf_high = df_4h['high'].values
    htf_low = df_4h['low'].values
    
    _, htf_trend_dir = calculate_supertrend(htf_high, htf_low, htf_close, period=10, multiplier=3.0)
    
    # Align 4h trend to 1h timeframe (auto shift(1) for completed bars)
    htf_trend_aligned = align_htf_to_ltf(prices, df_4h, htf_trend_dir)
    
    # Calculate 1h indicators
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bbw = calculate_bbw(close, high, low, period=20)
    
    # Calculate BBW percentile for regime filter
    # Only trade when BBW is above 50th percentile (expanding volatility)
    bbw_valid = bbw[50:]
    bbw_valid = bbw_valid[~np.isnan(bbw_valid)]
    bbw_median = np.median(bbw_valid) if len(bbw_valid) > 0 else 0.0
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_LONG = 0.30   # 30% position size for long
    SIZE_SHORT = -0.30 # 30% position size for short
    SIZE_HALF = 0.15   # Half position for take profit
    
    # Track position for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    r_multiple = 0.0   # Risk multiple for take profit
    
    # Minimum bars for all indicators to be valid
    min_bars = 50  # Ensure all indicators have enough data
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if np.isnan(htf_trend_aligned[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bbw[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when BBW above median (trending market)
        in_trending_regime = bbw[i] > bbw_median
        
        # 4h Supertrend direction
        htf_trend = htf_trend_aligned[i]
        
        # MACD histogram momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        
        # MACD histogram turning (early signal)
        macd_turning_long = macd_hist[i] > macd_hist[i-1] and macd_hist[i-1] < 0 if i > 0 else False
        macd_turning_short = macd_hist[i] < macd_hist[i-1] and macd_hist[i-1] > 0 if i > 0 else False
        
        # RSI pullback filter (enter on pullback within trend)
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Pullback in downtrend
        
        # Generate entry signals
        new_signal = 0.0
        
        if in_trending_regime:
            # Long entry: 4h bullish + MACD turning + RSI pullback
            if htf_trend == 1 and macd_turning_long and rsi_oversold:
                new_signal = SIZE_LONG
            
            # Short entry: 4h bearish + MACD turning + RSI pullback
            elif htf_trend == -1 and macd_turning_short and rsi_overbought:
                new_signal = SIZE_SHORT
        
        # Stoploss and Take Profit logic
        if position_side == 1 and new_signal > 0:  # Long position active
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr[i]
            
            # Calculate R multiple for take profit
            if entry_price > 0:
                r_multiple = (close[i] - entry_price) / (2.5 * atr[i]) if atr[i] > 0 else 0
            
            # Take profit at 2R: reduce to half position
            if r_multiple >= 2.0 and new_signal == SIZE_LONG:
                new_signal = SIZE_HALF
            
            # Stoploss hit
            if close[i] < stop_price:
                new_signal = 0.0
        
        elif position_side == -1 and new_signal < 0:  # Short position active
            lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr[i]
            
            # Calculate R multiple for take profit
            if entry_price > 0:
                r_multiple = (entry_price - close[i]) / (2.5 * atr[i]) if atr[i] > 0 else 0
            
            # Take profit at 2R: reduce to half position
            if r_multiple >= 2.0 and new_signal == SIZE_SHORT:
                new_signal = -SIZE_HALF
            
            # Stoploss hit
            if close[i] > stop_price:
                new_signal = 0.0
        
        # Update position tracking
        if signals[i-1] == 0 and new_signal != 0:
            # New entry
            position_side = 1 if new_signal > 0 else -1
            entry_price = close[i]
            highest_price = high[i]
            lowest_price = low[i]
            r_multiple = 0.0
        elif signals[i-1] != 0 and new_signal == 0:
            # Exit
            position_side = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            r_multiple = 0.0
        elif signals[i-1] != 0 and new_signal != 0 and position_side != (1 if new_signal > 0 else -1):
            # Position flip
            position_side = 1 if new_signal > 0 else -1
            entry_price = close[i]
            highest_price = high[i]
            lowest_price = low[i]
            r_multiple = 0.0
        
        signals[i] = new_signal
    
    return signals