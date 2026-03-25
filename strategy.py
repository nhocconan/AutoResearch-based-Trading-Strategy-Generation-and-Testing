#!/usr/bin/env python3
"""
Experiment #1432: 12h Primary + 1d HTF — Triple Trend Confirmation + RSI Pullback

Hypothesis: 12h timeframe offers optimal trade frequency (20-50/year) with lower fee drag.
This strategy combines THREE trend confirmations for robust signal generation:
1. 1d HMA(21) for major trend bias (regime filter)
2. 12h HMA(16/48) crossover for momentum direction
3. 12h KAMA(21) as volatility-adaptive trend confirmation
4. RSI(14) pullback entry in 35-65 zone (LOOSE to guarantee trades)
5. ATR(14) trailing stoploss (2.5x)

Why this should beat the baseline (Sharpe=0.575):
- 12h TF = fewer false signals than 4h/6h, more trades than 1d
- Triple trend confirmation (HMA + KAMA + 1d bias) reduces whipsaw
- KAMA adapts to volatility = better in choppy 2022-2023 period
- RSI 35-65 is LOOSE enough to trigger on normal pullbacks (not just extremes)
- Discrete sizing minimizes fee churn on signal changes

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1d_HMA bullish + 12h_HMA16 > 12h_HMA48 + 12h_KAMA rising + RSI > 35
- SHORT: 1d_HMA bearish + 12h_HMA16 < 12h_HMA48 + 12h_KAMA falling + RSI < 65

Timeframe: 12h
Size: 0.25 base, 0.30 strong (discrete)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_triple_trend_kama_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = np.nansum(np.abs(np.diff(close[max(0, i-period):i+1])))
            if noise > 0:
                er[i] = signal / noise
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]) and not np.isnan(kama[i-1]) and not np.isnan(close[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    kama_21 = calculate_kama(close, period=10, fast=2, slow=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER (trend momentum) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === KAMA TREND CONFIRMATION (volatility-adaptive) ===
        kama_rising = False
        kama_falling = False
        if i >= 2 and not np.isnan(kama_21[i-1]):
            kama_rising = kama_21[i] > kama_21[i-1]
            kama_falling = kama_21[i] < kama_21[i-1]
        
        # === RSI PULLBACK (LOOSE entry - guarantee trades) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 12h HMA bullish + KAMA rising + RSI > 35 (pullback zone)
        if price_above_1d and hma_bullish and kama_rising and rsi > 35:
            # Strong if RSI also < 70 (not overbought)
            if rsi < 70:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 12h HMA bearish + KAMA falling + RSI < 65 (pullback zone)
        elif price_below_1d and hma_bearish and kama_falling and rsi < 65:
            # Strong if RSI also > 30 (not oversold)
            if rsi > 30:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals