#!/usr/bin/env python3
"""
Experiment #070: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Filter

Hypothesis: 1h timeframe with HTF trend bias can work IF we use loose enough entry
conditions to generate 30-80 trades/year. Key insight from failures: too many filters
= 0 trades. This strategy uses:
1. 4h HMA(21) for trend direction (simple, proven)
2. 1h RSI(14) pullback entries (30/70 thresholds, not extreme 20/80)
3. Volume filter (>0.7x 20-bar avg) - loose enough to pass most bars
4. ATR stoploss (2.5x) to limit drawdown

Why this should work:
- 4h HMA provides clear trend bias without whipsaw
- RSI 30/70 triggers frequently enough (unlike 20/80)
- Volume filter is permissive (0.7x not 1.5x)
- Discrete sizing (0.25) limits drawdown during 2022 crash

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_vol_4h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # WMA(period/2)
    wma_half = close_series.rolling(window=period//2, min_periods=period//2).mean()
    
    # WMA(period)
    wma_full = close_series.rolling(window=period, min_periods=period).mean()
    
    # 2*WMA(period/2) - WMA(period)
    hull_raw = 2.0 * wma_half - wma_full
    
    # WMA(sqrt(period)) on the result
    sqrt_period = int(np.sqrt(period))
    hma = hull_raw.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for volatility filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized bandwidth
    
    return upper, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 12h HMA for secondary HTF confirmation
    df_12h = get_htf_data(prices, '12h')
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 12h CONFIRMATION ===
        htf2_bull = close[i] > hma_12h_aligned[i]
        htf2_bear = close[i] < hma_12h_aligned[i]
        
        # === 1h LOCAL TREND (HMA slope) ===
        hma_slope_bull = hma_1h[i] > hma_1h[i-5] if i >= 105 else False
        hma_slope_bear = hma_1h[i] < hma_1h[i-5] if i >= 105 else False
        
        # === RSI PULLBACK (loose thresholds for trade frequency) ===
        rsi_oversold = rsi[i] < 40.0  # Not too extreme
        rsi_overbought = rsi[i] > 60.0  # Not too extreme
        
        # === VOLUME FILTER (permissive) ===
        volume_ok = volume[i] > 0.7 * vol_ma[i]
        
        # === BOLLINGER BAND POSITION ===
        near_lower = close[i] < bb_lower[i] * 1.01  # At or below lower band
        near_upper = close[i] > bb_upper[i] * 0.99  # At or above upper band
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI pullback + volume + near BB lower
        if htf_bull and rsi_oversold and volume_ok:
            # At least 2 of 3: htf2_bull, hma_slope_bull, near_lower
            confirm_count = sum([htf2_bull, hma_slope_bull, near_lower])
            if confirm_count >= 2:
                desired_signal = SIZE_LONG
        
        # SHORT: HTF bear + RSI overbought + volume + near BB upper
        elif htf_bear and rsi_overbought and volume_ok:
            # At least 2 of 3: htf2_bear, hma_slope_bear, near_upper
            confirm_count = sum([htf2_bear, hma_slope_bear, near_upper])
            if confirm_count >= 2:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.85:
            final_signal = SIZE_LONG
        elif desired_signal <= -SIZE_SHORT * 0.85:
            final_signal = -SIZE_SHORT
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals