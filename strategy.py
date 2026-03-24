#!/usr/bin/env python3
"""
Experiment #060: 1h Primary + 4h/12h HTF — Regime-Adaptive Pullback Strategy

Hypothesis: 1h timeframe with strict HTF filtering can beat 4h baseline by:
1. Using 4h HMA for TREND DIRECTION (not entry)
2. Using 1h RSI for ENTRY TIMING (pullback within HTF trend)
3. Using Choppiness Index to AVOID whipsaw regimes
4. Using volume + session filters to reduce false signals
5. Target: 40-80 trades/year (strict enough for 1h, loose enough for trade gen)

Key lessons from failures:
- #048/#058: 0 trades = auto-reject (conditions too strict)
- #050/#055: Negative Sharpe on BTC/ETH (trend-only fails in 2022/2025)
- #052: Worked but low Sharpe (12h too slow)

This strategy uses OR logic for entries (multiple paths) to ensure trade generation
while keeping filters strict enough to limit trade frequency on 1h.

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if highest > lowest and atr_sum > 0:
            chop[i] = 100.0 * np.log10((highest - lowest) / atr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h/12h HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size - conservative for 1h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(vol_sma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC for liquid hours) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (> 0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === HTF BIAS (4h and 12h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === 1h TREND (HMA) ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === REGIME (Choppiness) ===
        is_choppy = chop_1h[i] > 55.0  # Range market
        is_trending = chop_1h[i] < 45.0  # Trend market
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi_1h[i] < 45.0  # Loose for trade generation
        rsi_overbought = rsi_1h[i] > 55.0  # Loose for trade generation
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # PATH 1: TREND FOLLOWING (trending regime + HTF alignment)
        if is_trending and in_session and volume_ok:
            # Long: 4h bull + 12h bull + 1h bull + RSI not overbought
            if hma_4h_bull and hma_12h_bull and hma_1h_bull and rsi_1h[i] < 65.0:
                desired_signal = SIZE
            # Short: 4h bear + 12h bear + 1h bear + RSI not oversold
            elif hma_4h_bear and hma_12h_bear and hma_1h_bear and rsi_1h[i] > 35.0:
                desired_signal = -SIZE
        
        # PATH 2: MEAN REVERSION (choppy regime + HTF bias + RSI extreme)
        if is_choppy and in_session and volume_ok:
            # Long: HTF bull + RSI oversold pullback
            if (hma_4h_bull or hma_12h_bull) and rsi_oversold:
                desired_signal = SIZE
            # Short: HTF bear + RSI overbought pullback
            elif (hma_4h_bear or hma_12h_bear) and rsi_overbought:
                desired_signal = -SIZE
        
        # PATH 3: STRONG TREND BREAKOUT (all HTF aligned, ignore RSI)
        if in_session and volume_ok:
            # Very strong long signal
            if hma_4h_bull and hma_12h_bull and hma_1h_bull and rsi_1h[i] > 50.0 and rsi_1h[i] < 70.0:
                desired_signal = SIZE
            # Very strong short signal
            elif hma_4h_bear and hma_12h_bear and hma_1h_bear and rsi_1h[i] < 50.0 and rsi_1h[i] > 30.0:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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