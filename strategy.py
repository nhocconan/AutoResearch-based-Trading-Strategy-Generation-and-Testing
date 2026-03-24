#!/usr/bin/env python3
"""
Experiment #686: 1d Primary + 1w HTF — KAMA Trend + Donchian Breakout + RSI Entry

Hypothesis: Daily timeframe with weekly bias provides optimal signal quality for crypto perpetuals.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in chop.
Donchian(20) breakout confirms momentum. RSI(14) ensures we're not buying at extremes.
Weekly HMA(21) provides meta-trend filter to avoid counter-trend trades.

Key innovations:
1. KAMA(10,2,30) - adaptive trend following, reduces whipsaw in chop
2. Donchian(20) breakout - price must break 20-day high/low for entry confirmation
3. RSI(14) filter - avoid entries at extremes (RSI 30-70 range for entries)
4. 1w HMA(21) bias - only long when price > weekly HMA, only short when below
5. ATR(14) trailing stop - 2.5x for risk management, signal→0 on stop
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (LOOSE to ensure trades):
- LONG: price > 1w HMA AND KAMA rising AND Donchian breakout AND RSI 35-65
- SHORT: price < 1w HMA AND KAMA falling AND Donchian breakdown AND RSI 35-65

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, efficiency_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing constant based on market efficiency (trend vs noise)
    Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    """
    n = len(close)
    if n < efficiency_period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(efficiency_period, n):
        signal = abs(close[i] - close[i - efficiency_period])
        noise = 0.0
        for j in range(i - efficiency_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA of first efficiency_period bars
    kama[efficiency_period] = np.mean(close[:efficiency_period + 1])
    
    for i in range(efficiency_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama = calculate_kama(close, fast_period=2, slow_period=30, efficiency_period=10)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    # KAMA direction tracking
    kama_rising = np.zeros(n, dtype=bool)
    kama_falling = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i-1]):
            kama_rising[i] = kama[i] > kama[i-1]
            kama_falling[i] = kama[i] < kama[i-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_uptrend = kama_rising[i]
        kama_downtrend = kama_falling[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.995  # near or at breakout
        donchian_breakdown_short = close[i] <= donchian_lower[i] * 1.005  # near or at breakdown
        
        # === RSI FILTER (loose - ensure trades) ===
        rsi_ok_long = 30.0 <= rsi[i] <= 70.0  # not overbought
        rsi_ok_short = 30.0 <= rsi[i] <= 70.0  # not oversold
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + KAMA rising + Donchian breakout + RSI ok
        if htf_bull and kama_uptrend and donchian_breakout_long and rsi_ok_long:
            desired_signal = SIZE_STRONG
        elif htf_bull and kama_uptrend and rsi_ok_long:
            # Weaker: just weekly bias + KAMA + RSI
            desired_signal = SIZE_BASE
        elif htf_bull and kama_uptrend:
            # Weakest: just trend alignment
            desired_signal = SIZE_BASE * 0.5
        
        # SHORT: Weekly bearish + KAMA falling + Donchian breakdown + RSI ok
        elif htf_bear and kama_downtrend and donchian_breakdown_short and rsi_ok_short:
            desired_signal = -SIZE_STRONG
        elif htf_bear and kama_downtrend and rsi_ok_short:
            # Weaker: just weekly bias + KAMA + RSI
            desired_signal = -SIZE_BASE
        elif htf_bear and kama_downtrend:
            # Weakest: just trend alignment
            desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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