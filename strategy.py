#!/usr/bin/env python3
"""
Experiment #045: 1h Primary + 4h/1d HTF — Session-Filtered HMA + RSI Pullback

Hypothesis: After 44 experiments, lower TF (1h) strategies fail due to:
1. Too many trades → fee drag (need 30-60/year, not 200+)
2. Too strict entry filters → 0 trades (Sharpe=0.000)
3. No session filtering → trades during low-liquidity Asian hours

Solution: Combine PROVEN elements from #043 (1d HMA trend) with STRICT filters:
1. 1d HMA for major trend bias (proven Sharpe=0.313 in #043)
2. 4h HMA for intermediate trend confirmation
3. 1h RSI pullback entries with LOOSE thresholds (45/55 not 30/70)
4. SESSION FILTER: Only trade 8-20 UTC (London/NY overlap = high liquidity)
5. VOLUME FILTER: Volume > 1.0x 20-bar average (confirm participation)
6. ATR FILTER: ATR > 0.8x 50-bar avg (avoid dead markets)

Key insight: STRICT filters on WHEN to trade + LOOSE filters on ENTRY trigger
= Few trades (30-60/year) but high quality entries with HTF trend.

Size: 0.25 (smaller for 1h to reduce fee impact from more frequent signals)
Target: Beat Sharpe=0.313, trades>120/symbol train, >12/symbol test, DD>-40%
Timeframe: 1h (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_session_volume_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds for trade generation"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss and volume filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_session_hour(prices):
    """Extract UTC hour from open_time for session filtering"""
    # open_time is in milliseconds since epoch
    hours = (prices['open_time'].values // 3600000) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_avg_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Session hours (UTC)
    session_hours = calculate_session_hour(prices)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]) or np.isnan(atr_avg_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only - London/NY overlap) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === VOLUME FILTER (>1.0x 20-bar average) ===
        volume_ok = volume[i] > 1.0 * vol_avg_20[i]
        
        # === ATR FILTER (>0.8x 50-bar average - avoid dead markets) ===
        atr_ok = atr[i] > 0.8 * atr_avg_50[i]
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h TREND ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === DESIRED SIGNAL (LOOSE RSI thresholds, STRICT filters) ===
        desired_signal = 0.0
        
        # LONG: All HTF bullish + session + volume + atr + RSI pullback
        # Requires: 1d bull + 4h bull + session + volume_ok + atr_ok + RSI < 55
        if hma_1d_bull and hma_4h_bull:
            if in_session and volume_ok and atr_ok:
                if rsi[i] < 55.0:
                    # Pullback in uptrend - buy the dip (loose threshold)
                    desired_signal = SIZE
        
        # SHORT: All HTF bearish + session + volume + atr + RSI rally
        # Requires: 1d bear + 4h bear + session + volume_ok + atr_ok + RSI > 45
        if hma_1d_bear and hma_4h_bear:
            if in_session and volume_ok and atr_ok:
                if rsi[i] > 45.0:
                    # Rally in downtrend - sell the rip (loose threshold)
                    desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                # Flip position
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