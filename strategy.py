#!/usr/bin/env python3
"""
EXPERIMENT #012 - KAMA Adaptive Trend + BB Squeeze + RSI Momentum
=================================================================
Hypothesis: KAMA adapts to market noise better than HMA/EMA - moves fast in trends,
slow in chop. Combined with BB squeeze (volatility contraction before expansion)
and RSI momentum filter, this should capture breakouts with better risk control.

Key improvements over mtf_hma_rsi_zscore_v1:
- KAMA instead of HMA (adaptive to volatility, less whipsaw)
- BB squeeze filter (volatility regime detection - trade before expansion)
- ATR-based stoploss (signal→0 when 2*ATR against position)
- Conservative sizing (0.25-0.35 max) to control drawdown

Why this might beat Sharpe=1.768:
- KAMA reduces whipsaw in choppy markets (ER adapts smoothing)
- BB squeeze captures volatility expansion breakouts
- ATR stops limit drawdown on failed breakouts
- Multi-timeframe already proven to work
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bb_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period:
        return kama
    
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Initialize KAMA with SMA of first period
    kama[period-1] = np.mean(close[:period])
    
    for i in range(period, n):
        # Efficiency Ratio
        signal = abs(close[i] - close[i-period])
        noise = 0.0
        for j in range(i-period+1, i+1):
            noise += abs(close[j] - close[j-1])
        
        if noise == 0:
            er = 0.0
        else:
            er = signal / noise
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_bollinger_bands(close, period=20, std_mult=2):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / np.where(middle != 0, middle, 1)
    
    return upper, lower, middle, bandwidth, std


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
    rsi[avg_loss == 0] = 100.0
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_middle, bb_width, bb_std = calculate_bollinger_bands(close, period=20)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(df_4h['close'].values, period=10)
    
    # Calculate 4h trend direction (price vs KAMA)
    trend_4h = np.zeros(len(kama_4h))
    close_4h = df_4h['close'].values
    for i in range(10, len(kama_4h)):
        if kama_4h[i] > 0:
            if close_4h[i] > kama_4h[i]:
                trend_4h[i] = 1  # Bullish
            else:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    n_4h = len(df_4h)
    
    for i in range(n):
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h >= 10:
            trend_1h[i] = trend_4h[idx_4h]
    
    # BB width percentile for squeeze detection (rolling 100 bars)
    bb_percentile = np.zeros(n)
    window = 100
    for i in range(window, n):
        past_widths = bb_width[i-window:i]
        bb_percentile[i] = np.sum(past_widths < bb_width[i]) / window
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # RSI thresholds for momentum entries
    RSI_LONG_MIN = 45   # Minimum RSI for long entry
    RSI_LONG_MAX = 65   # Maximum RSI for long entry (not overbought)
    RSI_SHORT_MIN = 35  # Minimum RSI for short entry (not oversold)
    RSI_SHORT_MAX = 55  # Maximum RSI for short entry
    
    # BB squeeze threshold (bottom 30% of historical width = low volatility)
    BB_SQUEEZE_THRESHOLD = 0.30
    
    # Track position for stoploss
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=flat
    entry_price = np.zeros(n)
    long_stop = np.zeros(n)  # Highest close since long entry
    short_stop = np.zeros(n)  # Lowest close since short entry
    
    first_valid = max(48, 20, 14, 100)  # Wait for all indicators
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            position_direction[i] = 0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_pct = bb_percentile[i]
        atr_val = atr_1h[i] if atr_1h[i] > 0 else close[i] * 0.01
        
        # Initialize tracking from previous bar
        position_direction[i] = position_direction[i-1]
        entry_price[i] = entry_price[i-1]
        long_stop[i] = long_stop[i-1]
        short_stop[i] = short_stop[i-1]
        
        # Update stops based on current position
        if position_direction[i-1] == 1:  # Was long
            long_stop[i] = max(long_stop[i-1], close[i-1])
        elif position_direction[i-1] == -1:  # Was short
            if short_stop[i-1] == 0:
                short_stop[i] = close[i-1]
            else:
                short_stop[i] = min(short_stop[i-1], close[i-1])
        
        # Check ATR stoploss - exit if 2*ATR against position
        if position_direction[i-1] == 1:  # Long position
            if close[i] < long_stop[i] - 2 * atr_val:
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
                long_stop[i] = 0
                continue
        elif position_direction[i-1] == -1:  # Short position
            if short_stop[i-1] > 0 and close[i] > short_stop[i] + 2 * atr_val:
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
                short_stop[i] = 0
                continue
        
        # BB squeeze filter - only trade when volatility is low (before expansion)
        squeeze_active = bb_pct < BB_SQUEEZE_THRESHOLD
        
        if trend == 1:  # 4h uptrend
            if rsi_val > RSI_LONG_MIN and rsi_val < RSI_LONG_MAX:
                if squeeze_active:
                    signals[i] = SIZE_FULL
                else:
                    signals[i] = SIZE_HALF
            else:
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_MIN and rsi_val < RSI_SHORT_MAX:
                if squeeze_active:
                    signals[i] = -SIZE_FULL
                else:
                    signals[i] = -SIZE_HALF
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
        
        # Update position tracking
        if signals[i] != 0 and position_direction[i] == 0:
            # New position
            position_direction[i] = np.sign(signals[i])
            entry_price[i] = close[i]
            if signals[i] > 0:
                long_stop[i] = close[i]
            else:
                short_stop[i] = close[i]
        elif signals[i] == 0:
            position_direction[i] = 0
            entry_price[i] = 0
            long_stop[i] = 0
            short_stop[i] = 0
    
    return signals