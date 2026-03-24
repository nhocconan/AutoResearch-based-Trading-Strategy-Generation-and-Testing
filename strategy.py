#!/usr/bin/env python3
"""
Experiment #655: 6h Primary + 12h/1d HTF — Dual HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: 6h timeframe is the "sweet spot" between 4h (too many trades/fees) and 12h (too few trades).
Using 12h HMA for primary trend direction + 1d HMA for confirmation creates a robust trend filter.
RSI(14) pullback entries in trend direction capture better risk/reward than pure breakouts.
Donchian(20) breakout adds momentum confirmation when trend accelerates.

Key innovations:
1. 12h HMA(21) primary trend - faster than 1d, captures multi-day swings
2. 1d HMA(21) secondary filter - confirms long-term direction
3. RSI(14) pullback entry - buy dips in uptrend, sell rallies in downtrend
4. Donchian(20) breakout - captures momentum when trend accelerates
5. Dual entry modes - pullback (RSI) OR breakout (Donchian) = more trades
6. ATR(14) trailing stop 2.5x - protects capital in 2022-style crashes

Entry conditions (LOOSE to ensure >=30 trades/train, >=3/test):
- LONG: 12h HMA bull OR 1d HMA bull + (RSI<45 pullback OR Donchian breakout)
- SHORT: 12h HMA bear OR 1d HMA bear + (RSI>55 rally OR Donchian breakdown)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-30%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_hma_rsi_donchian_12h1d_v1"
timeframe = "6h"
leverage = 1.0

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
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0  # No loss = RSI 100
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss <= 1e-10] = 100.0
    
    return rsi

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
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian midpoint for pullback reference
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]):
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h + 1d HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Combined bias: at least one HTF agrees
        htf_bull = htf_12h_bull or htf_1d_bull
        htf_bear = htf_12h_bear or htf_1d_bear
        
        # Strong bias: both HTF agree
        htf_strong_bull = htf_12h_bull and htf_1d_bull
        htf_strong_bear = htf_12h_bear and htf_1d_bear
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi[i] < 45.0  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55.0  # Rally in downtrend
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] >= donchian_upper[i]
        breakout_short = close[i] <= donchian_lower[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS - ensure trades) ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG entries (multiple pathways to ensure trades)
        if htf_bull:
            # Path 1: RSI pullback in uptrend
            if rsi_oversold:
                desired_signal = SIZE_BASE
                signal_strength = 1
            # Path 2: Donchian breakout with HTF bull
            if breakout_long:
                desired_signal = max(desired_signal, SIZE_BASE)
                signal_strength = max(signal_strength, 1)
            # Path 3: Strong HTF + RSI pullback
            if htf_strong_bull and rsi_oversold:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            # Path 4: Strong HTF + Donchian breakout
            if htf_strong_bull and breakout_long:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            # Path 5: Extreme RSI oversold (strong mean reversion)
            if rsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_STRONG
                signal_strength = 2
        
        # SHORT entries (multiple pathways to ensure trades)
        if htf_bear:
            # Path 1: RSI rally in downtrend
            if rsi_overbought:
                desired_signal = min(desired_signal, -SIZE_BASE) if desired_signal <= 0 else -SIZE_BASE
                signal_strength = max(signal_strength, 1)
            # Path 2: Donchian breakdown with HTF bear
            if breakout_short:
                if desired_signal >= 0:
                    desired_signal = -SIZE_BASE
                signal_strength = max(signal_strength, 1)
            # Path 3: Strong HTF + RSI rally
            if htf_strong_bear and rsi_overbought:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            # Path 4: Strong HTF + Donchian breakdown
            if htf_strong_bear and breakout_short:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            # Path 5: Extreme RSI overbought (strong mean reversion)
            if rsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
        
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
            signal_strength = 0
        
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