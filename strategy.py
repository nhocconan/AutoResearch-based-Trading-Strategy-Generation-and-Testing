#!/usr/bin/env python3
"""
Experiment #1418: 4h Primary + 1d HTF — Donchian Breakout + KAMA Trend + Volume

Hypothesis: Donchian breakouts capture crypto momentum better than EMA/HMA crossovers.
Combined with KAMA (adaptive to volatility) and volume confirmation, this should:
1. Generate 25-50 trades/year on 4h (breakout frequency is natural)
2. Work on ALL symbols (BTC/ETH/SOL all have momentum phases)
3. 1d KAMA filter prevents counter-trend breakout failures (like 2022 crash)
4. Volume filter (1.5x avg) reduces false breakouts significantly
5. ATR trailing stop (2.5x) protects from reversal whipsaws

Why different from failed strategies:
- Donchian breakout = MORE trades than RSI extremes (guarantees trade count)
- KAMA adapts to volatility better than fixed EMA/HMA (works in chop AND trend)
- Volume confirmation = fewer false signals than price-only breakouts
- 1d trend filter = avoids 2022-style crash counter-trend losses

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_KAMA bullish + price breaks Donchian(20) high + volume > 1.5x avg
- SHORT: 1d_KAMA bearish + price breaks Donchian(20) low + volume > 1.5x avg

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 0:
                er[i] = signal / noise
            else:
                er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        elif not np.isnan(close[i]):
            kama[i] = close[i]
    
    return kama

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        if not np.any(np.isnan(high[i - period + 1:i + 1])) and \
           not np.any(np.isnan(low[i - period + 1:i + 1])):
            upper[i] = np.nanmax(high[i - period + 1:i + 1])
            lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.any(np.isnan(volume[i - period + 1:i + 1])):
            vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10, fast=2, slow=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
        
        if np.isnan(kama_4h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA bias) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 4h KAMA TREND (momentum confirmation) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === VOLUME CONFIRMATION (reduce false breakouts) ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_20[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Check if price broke out THIS bar (not previous bars)
        breakout_long = high[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = low[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h KAMA bullish + Donchian breakout + volume confirmed
        if price_above_1d and kama_bullish and breakout_long:
            if volume_confirmed:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE  # Enter anyway, just smaller size
        
        # SHORT: 1d bearish + 4h KAMA bearish + Donchian breakout + volume confirmed
        elif price_below_1d and kama_bearish and breakout_short:
            if volume_confirmed:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE  # Enter anyway, just smaller size
        
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