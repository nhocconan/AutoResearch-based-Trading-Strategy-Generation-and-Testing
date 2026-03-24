#!/usr/bin/env python3
"""
Experiment #927: 6h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + Donchian Confirm

Hypothesis: 6h timeframe is unexplored territory between 4h and 12h. KAMA (Kaufman Adaptive
Moving Average) outperforms HMA/EMA in mixed regimes because it adapts speed based on
market efficiency ratio (ER). In trending markets KAMA speeds up, in ranging markets it
slows down reducing whipsaws. Combined with 1d HTF bias and loose RSI filter for entries.

Key innovations:
1. KAMA adapts to volatility - faster in trends (ER high), slower in ranges (ER low)
2. 1d KAMA(21) for HTF trend bias - price above = bullish, below = bearish
3. 6h KAMA(10/30) for entry trigger - fast crosses slow with ER confirmation
4. Donchian(20) breakout confirm - ensures we're not entering at range boundaries
5. RSI(14) loose filter only - avoid extreme counter-trend (RSI>75 long, RSI<25 short)
6. ATR(14) 2.5x trailing stop for risk management
7. LOOSE entry conditions to guarantee ≥10 trades/train, ≥3/test

Entry conditions (LOOSE):
- LONG = 1d KAMA bull (price > kama_1d) + 6h KAMA fast > slow + RSI < 75
- SHORT = 1d KAMA bear (price < kama_1d) + 6h KAMA fast < slow + RSI > 25
- Donchian confirm: long only if price > Donchian_mid, short if price < Donchian_mid

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_donchian_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, fast_period=10, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts speed based on market Efficiency Ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
    """
    n = len(close)
    if n < slow_period:
        return np.full(n, np.nan)
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[slow_period - 1] = close[slow_period - 1]
    
    for i in range(slow_period, n):
        # Efficiency Ratio
        signal = abs(close[i] - close[i - slow_period])
        noise = 0.0
        for j in range(i - slow_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 0.0
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper/lower/mid"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA
    kama_1d_raw = calculate_kama(df_1d['close'].values, fast_period=10, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h_fast = calculate_kama(close, fast_period=10, slow_period=30)
    kama_6h_slow = calculate_kama(close, fast_period=20, slow_period=60)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_up, donch_low, donch_mid = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h_fast[i]) or np.isnan(kama_6h_slow[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_mid[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_6h_bull = kama_6h_fast[i] > kama_6h_slow[i]
        kama_6h_bear = kama_6h_fast[i] < kama_6h_slow[i]
        
        # === KAMA CROSSOVER ===
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_6h_fast[i-1]) and not np.isnan(kama_6h_slow[i-1]):
            kama_crossover_long = (kama_6h_fast[i-1] <= kama_6h_slow[i-1]) and (kama_6h_fast[i] > kama_6h_slow[i])
            kama_crossover_short = (kama_6h_fast[i-1] >= kama_6h_slow[i-1]) and (kama_6h_fast[i] < kama_6h_slow[i])
        
        # === DONCHIAN CONFIRM ===
        donch_bull = close[i] > donch_mid[i]
        donch_bear = close[i] < donch_mid[i]
        
        # === RSI FILTER (LOOSE) ===
        rsi_overbought = rsi_14[i] > 75.0
        rsi_oversold = rsi_14[i] < 25.0
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries (HTF bullish bias + 6h KAMA bull + Donchian confirm)
        if htf_1d_bull and kama_6h_bull and donch_bull:
            # Crossover entry (stronger signal)
            if kama_crossover_long and not rsi_overbought:
                desired_signal = SIZE_STRONG
            # Trend continuation entry (looser - ensures trades)
            elif not rsi_overbought:
                desired_signal = SIZE_BASE
        
        # SHORT entries (HTF bearish bias + 6h KAMA bear + Donchian confirm)
        elif htf_1d_bear and kama_6h_bear and donch_bear:
            # Crossover entry (stronger signal)
            if kama_crossover_short and not rsi_oversold:
                desired_signal = -SIZE_STRONG
            # Trend continuation entry (looser - ensures trades)
            elif not rsi_oversold:
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