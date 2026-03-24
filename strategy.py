#!/usr/bin/env python3
"""
Experiment #663: 6h Primary + 1d/1w HTF — RSI Pullback + Donchian Momentum + Weekly Bias

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy uses:
1. 1w HMA(21) for major trend bias (very slow, avoids whipsaw)
2. 1d HMA(21) for intermediate trend confirmation
3. 6h RSI(14) for pullback entries (oversold in uptrend, overbought in downtrend)
4. 6h Donchian(10) for momentum confirmation (price in upper/lower half)
5. ATR(14) trailing stop at 2.0x for risk management

Key insight from failures: TOO MANY filters = 0 trades. This strategy uses LOOSE conditions:
- RSI thresholds: <40 for long, >60 for short (not extreme 30/70)
- Donchian: just needs to be in correct half (not breakout)
- HTF: only for direction bias, not hard filter
- This should generate 30-50 trades/year on 6h

Target: Sharpe>0.40 (beat current best 0.399), trades>=30 train, trades>=3 test
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi_pullback_donchian_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match original length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    mask = avg_loss > 1e-10
    rs = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs))
    rsi[~mask & (avg_gain > 0)] = 100.0
    
    return rsi

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_donchian(high, low, period=10):
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=10)
    atr = calculate_atr(high, low, close, period=14)
    
    # Calculate Donchian midpoint
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        # === HTF BIAS (1w and 1d HMA) ===
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        
        # Weekly bias (major trend)
        htf_bull = False
        htf_bear = False
        if hma_1w_valid:
            htf_bull = close[i] > hma_1w_aligned[i]
            htf_bear = close[i] < hma_1w_aligned[i]
        elif hma_1d_valid:
            htf_bull = close[i] > hma_1d_aligned[i]
            htf_bear = close[i] < hma_1d_aligned[i]
        
        # Daily confirmation
        daily_bull = hma_1d_valid and close[i] > hma_1d_aligned[i]
        daily_bear = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi[i] < 40.0  # Loose threshold for more trades
        rsi_overbought = rsi[i] > 60.0  # Loose threshold for more trades
        
        # === DONCHIAN MOMENTUM ===
        donchian_bull = close[i] > donchian_mid[i]  # In upper half
        donchian_bear = close[i] < donchian_mid[i]  # In lower half
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + RSI oversold + Donchian bull
        if htf_bull and rsi_oversold and donchian_bull:
            if daily_bull:
                desired_signal = SIZE_STRONG  # All aligned
            else:
                desired_signal = SIZE_BASE  # Weekly only
        elif htf_bull and rsi_oversold:
            # Just weekly + RSI, no Donchian confirm
            desired_signal = SIZE_BASE * 0.5
        elif rsi_oversold and donchian_bull:
            # Just RSI + Donchian, no HTF
            desired_signal = SIZE_BASE * 0.5
        
        # SHORT: Weekly bear + RSI overbought + Donchian bear
        elif htf_bear and rsi_overbought and donchian_bear:
            if daily_bear:
                desired_signal = -SIZE_STRONG  # All aligned
            else:
                desired_signal = -SIZE_BASE  # Weekly only
        elif htf_bear and rsi_overbought:
            # Just weekly + RSI, no Donchian confirm
            desired_signal = -SIZE_BASE * 0.5
        elif rsi_overbought and donchian_bear:
            # Just RSI + Donchian, no HTF
            desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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