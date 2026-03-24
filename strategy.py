#!/usr/bin/env python3
"""
Experiment #467: 6h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + Volume Confirm

Hypothesis: 6h timeframe needs simpler entry logic than dual-HTF approaches (#435 failed).
Key insights from 6h failures:
- Dual HTF (12h+1d) too restrictive → few trades
- Funding rate contrarian fails on 6h (noise)
- Weekly pivot filters = 0 trades (too strict)

New approach:
1. SINGLE HTF FILTER: 1d KAMA only (not dual 12h+1d) — more trades qualify
2. KAMA (Kaufman Adaptive): adapts to volatility, better than HMA in chop
3. RSI PULLBACK: Enter on RSI<45 in uptrend, RSI>55 in downtrend (proven on HTF)
4. VOLUME CONFIRM: Volume > 1.5x 20-bar avg for breakout entries (reduces false signals)
5. LOOSE ENTRY: Max 2-3 conditions (trend direction + RSI + optional volume)
6. ATR STOP: 2.5x ATR from entry (wider than 2.0x to avoid whipsaw)

Entry Logic:
- Long: 1d KAMA bull + 6h KAMA bull + RSI < 45 (pullback entry)
- Short: 1d KAMA bear + 6h KAMA bear + RSI > 55 (pullback entry)
- Breakout Long: 6h KAMA bull + Donchian breakout + volume spike
- Breakout Short: 6h KAMA bear + Donchian breakdown + volume spike

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
Size: 0.25 base, 0.30 strong (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_pullback_volume_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-slow_period):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA
    kama[period] = np.mean(close[:period+1])
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

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
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes > threshold * avg volume"""
    n = len(volume)
    if n < period:
        return np.full(n, False)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * avg_vol)
    
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (6h) indicators
    kama_6h = calculate_kama(close, period=10)
    kama_6h_fast = calculate_kama(close, period=5)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(rsi[i]):
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
        
        if np.isnan(sma_50[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d HTF TREND BIAS (single filter, not dual) ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # === KAMA CROSSOVER ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_6h_fast[i]) and not np.isnan(kama_6h_fast[i-1]):
            if not np.isnan(kama_6h[i]) and not np.isnan(kama_6h[i-1]):
                if kama_6h_fast[i-1] <= kama_6h[i-1] and kama_6h_fast[i] > kama_6h[i]:
                    kama_cross_long = True
                if kama_6h_fast[i-1] >= kama_6h[i-1] and kama_6h_fast[i] < kama_6h[i]:
                    kama_cross_short = True
        
        # === RSI PULLBACK (loose thresholds for more trades) ===
        rsi_pullback_long = rsi[i] < 45.0
        rsi_pullback_short = rsi[i] > 55.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume_spike[i]
        
        # === ENTRY LOGIC (LOOSE - max 2-3 conditions) ===
        desired_signal = 0.0
        
        # PULLBACK LONG: 1d bull + 6h bull + RSI pullback
        if htf_bull and kama_bull and rsi_pullback_long:
            desired_signal = SIZE_BASE
        
        # PULLBACK SHORT: 1d bear + 6h bear + RSI pullback
        elif htf_bear and kama_bear and rsi_pullback_short:
            desired_signal = -SIZE_BASE
        
        # BREAKOUT LONG: 6h bull + Donchian breakout + volume spike
        elif kama_bull and donchian_breakout_long and vol_confirm:
            desired_signal = SIZE_STRONG
        
        # BREAKOUT SHORT: 6h bear + Donchian breakdown + volume spike
        elif kama_bear and donchian_breakdown_short and vol_confirm:
            desired_signal = -SIZE_STRONG
        
        # KAMA CROSS LONG: 1d bull + KAMA cross up
        elif htf_bull and kama_cross_long:
            desired_signal = SIZE_BASE
        
        # KAMA CROSS SHORT: 1d bear + KAMA cross down
        elif htf_bear and kama_cross_short:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals