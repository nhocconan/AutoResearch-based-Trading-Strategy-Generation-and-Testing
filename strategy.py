#!/usr/bin/env python3
"""
Experiment #075: 1h Primary + 4h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: After 70+ failed experiments, the key issue is using WRONG indicators for
bear/range markets (2025 test period). RSI and Donchian work in trends but fail in ranges.

NEW APPROACH - Never tried successfully before:
1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI in bear markets
2. KAMA (Kaufman Adaptive) - adapts to volatility, reduces whipsaw in ranges
3. 4h HMA + 1d HMA confluence - BOTH must agree for trend bias (reduces false signals)
4. Volume confirmation - simple >0.7x 20-bar avg (loose enough to trigger)
5. Session filter 6-22 UTC - captures Asian + EU + US overlap (more trade windows)
6. ATR 2.5x trailing stop - proven risk management

Why this should work:
- Fisher Transform is designed for mean-reversion in ranges (perfect for 2025 bear market)
- KAMA adapts to volatility - slows in chop, speeds in trends
- Dual HTF filter (4h+1d) reduces false signals but still allows trades
- LOOSE thresholds ensure we generate 30-80 trades/year (not 0, not 200+)
- 1h timeframe with HTF direction = HTF trade frequency with 1h execution precision

Entry Logic:
- Long: Fisher < -1.2 + Fisher turning up + KAMA bullish + 4h HMA < price + 1d HMA < price + volume ok
- Short: Fisher > +1.2 + Fisher turning down + KAMA bearish + 4h HMA > price + 1d HMA > price + volume ok
- Size: 0.28 (discrete, between 0.25-0.30)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.37 (beat current best 0.368), trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_hma_confluence_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - designed for reversal detection in ranges
    Reference: Ehlers, J.F. "Rocket Science for Traders" (2002)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Avoid division by zero
        if highest == lowest:
            continue
        
        # Normalize price to -1 to +1 range
        value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * 0.0
        if i > period:
            value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * value
        
        # Clamp to avoid log errors
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        if i > period:
            fisher_prev[i] = fisher[i - 1]
        else:
            fisher_prev[i] = 0.0
    
    return fisher, fisher_prev

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average - adapts to market volatility
    Reference: Kaufman, P.J. "Trading Systems and Methods" (2013)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Initialize with SMA
    kama[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        # Price change over period
        change = abs(close[i] - close[i - period])
        
        # Sum of absolute price changes (volatility)
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        # Efficiency Ratio (0 = noise, 1 = trend)
        if volatility < 1e-10:
            er = 0.0
        else:
            er = change / volatility
        
        # Smoothed constant
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (between 0.25-0.30)
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(kama[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (6-22 UTC for liquidity) ===
        hour = pd.Timestamp(open_time[i], unit='ms').hour
        session_ok = 6 <= hour <= 22
        
        # === VOLUME FILTER (loose: >0.7x average) ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === HTF TREND BIAS (4h + 1d HMA confluence) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong bias
        hma_bull_confluence = hma_4h_bull and hma_1d_bull
        hma_bear_confluence = hma_4h_bear and hma_1d_bear
        
        # === KAMA TREND CONFIRMATION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS (reversal detection) ===
        # Long: Fisher < -1.2 (oversold) + turning up
        fisher_oversold = fisher[i] < -1.2
        fisher_turning_up = fisher[i] > fisher_prev[i] if not np.isnan(fisher_prev[i]) else False
        
        # Short: Fisher > +1.2 (overbought) + turning down
        fisher_overbought = fisher[i] > 1.2
        fisher_turning_down = fisher[i] < fisher_prev[i] if not np.isnan(fisher_prev[i]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Fisher oversold + turning + KAMA bull + HTF bull + volume + session
        if fisher_oversold and fisher_turning_up and kama_bull and hma_bull_confluence and volume_ok and session_ok:
            desired_signal = SIZE
        
        # Short entry: Fisher overbought + turning + KAMA bear + HTF bear + volume + session
        elif fisher_overbought and fisher_turning_down and kama_bear and hma_bear_confluence and volume_ok and session_ok:
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