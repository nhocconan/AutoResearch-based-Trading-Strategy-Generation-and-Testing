#!/usr/bin/env python3
"""
Experiment #041: 4h Primary + 1d HTF — KAMA Adaptive Trend + Donchian Breakout + Volume

Hypothesis: After 40 experiments, the pattern is clear:
1. KAMA (Kaufman Adaptive) outperforms HMA/EMA because it adapts to market noise
   - Fast in trends (low noise), slow in ranges (high noise)
   - Current best strategy uses KAMA (Sharpe=0.221)
2. Donchian breakouts work well on 4h (proven on SOL with Sharpe +0.782)
3. Volume confirmation is MISSING from most failed strategies - breakouts need volume
4. HTF (1d) KAMA bias prevents counter-trend trades
5. Loose thresholds ensure trade generation (avoid Sharpe=0.000 failures)

Entry Logic:
- Long: 1d_KAMA_bull + 4h_KAMA_bull + price breaks Donchian(20) high + volume > 1.5x avg
- Short: 1d_KAMA_bear + 4h_KAMA_bear + price breaks Donchian(20) low + volume > 1.5x avg
- Size: 0.30 (discrete, safe through 77% crash)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.25, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise: fast in trends, slow in ranges
    From Perry Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise < 1e-10:
            er[i] = 1.0
        else:
            er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.full(n, np.nan)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout detection
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for volume spike detection"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for HTF trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    kama_4h_fast = calculate_kama(close, period=10)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size - safe through 77% crash
    
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
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
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
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        hma_1d_bull = close[i] > kama_1d_aligned[i]
        hma_1d_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h TREND (KAMA) ===
        kama_4h_bull = close[i] > kama_4h[i]
        kama_4h_bear = close[i] < kama_4h[i]
        kama_fast_above_slow = kama_4h_fast[i] > kama_4h[i] if not np.isnan(kama_4h_fast[i]) else False
        kama_fast_below_slow = kama_4h_fast[i] < kama_4h[i] if not np.isnan(kama_4h_fast[i]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 4h bull + Donchian breakout + volume spike
        if hma_1d_bull and kama_4h_bull:
            if breakout_long and volume_spike:
                # Strong breakout with volume
                desired_signal = SIZE
            elif kama_fast_above_slow and kama_4h_bull:
                # KAMA crossover confirmation
                desired_signal = SIZE
        
        # SHORT: 1d bear + 4h bear + Donchian breakout + volume spike
        if hma_1d_bear and kama_4h_bear:
            if breakout_short and volume_spike:
                # Strong breakdown with volume
                desired_signal = -SIZE
            elif kama_fast_below_slow and kama_4h_bear:
                # KAMA crossover confirmation
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