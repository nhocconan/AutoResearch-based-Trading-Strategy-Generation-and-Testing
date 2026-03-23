#!/usr/bin/env python3
"""
Experiment #1339: 4h Primary + 1d HTF — KAMA Trend + Donchian Breakout + RSI Momentum

Hypothesis: 4h timeframe with 1d HMA trend filter balances trade frequency (target 30-50/year)
with signal quality. KAMA adapts to volatility better than EMA/HMA. Donchian(20) breakouts
capture momentum moves. RSI(7) is faster than RSI(14) for entry timing. Simpler entry logic
than #1337 to ensure sufficient trades (avoiding 0-trade failure mode).

Key design choices:
1. 1d HMA(21) for macro trend bias - proven in #1337 (Sharpe=0.618)
2. KAMA(10,2,30) adapts to volatility - reduces whipsaw in chop
3. Donchian(20) breakout as trigger - captures momentum without lag
4. RSI(7) with moderate bands (35-65) - confirms without over-filtering
5. ATR(14) trailing stop 2.5x - tighter than 3x for better risk/reward
6. Position size 0.28 - conservative for 4h volatility

Target: 30-50 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < slow_period + period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        noise = 0.0
        for j in range(i - slow_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 4h entries"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # KAMA slope for trend confirmation
    kama_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i-1]):
            kama_slope[i] = kama[i] - kama[i-1]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (KAMA) ===
        kama_bull = (close[i] > kama[i]) and (kama_slope[i] > 0)
        kama_bear = (close[i] < kama[i]) and (kama_slope[i] < 0)
        
        # === RSI MOMENTUM (moderate bands for trade frequency) ===
        rsi_bull = rsi[i] > 45.0
        rsi_bear = rsi[i] < 55.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Simpler logic to ensure trades happen
        if macro_bull:
            # Path 1: Donchian breakout + RSI confirmation (primary)
            if breakout_long and rsi_bull:
                desired_signal = BASE_SIZE
            # Path 2: KAMA bull + RSI strong (trend follow)
            elif kama_bull and rsi_strong_bull:
                desired_signal = BASE_SIZE
            # Path 3: Price above both KAMA and 1d HMA (simple trend)
            elif close[i] > kama[i] and close[i] > hma_1d_aligned[i]:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY: Simpler logic to ensure trades happen
        elif macro_bear:
            # Path 1: Donchian breakout + RSI confirmation (primary)
            if breakout_short and rsi_bear:
                desired_signal = -BASE_SIZE
            # Path 2: KAMA bear + RSI strong (trend follow)
            elif kama_bear and rsi_strong_bear:
                desired_signal = -BASE_SIZE
            # Path 3: Price below both KAMA and 1d HMA (simple trend)
            elif close[i] < kama[i] and close[i] < hma_1d_aligned[i]:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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