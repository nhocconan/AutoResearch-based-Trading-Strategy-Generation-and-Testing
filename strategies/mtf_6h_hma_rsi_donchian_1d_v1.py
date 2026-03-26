#!/usr/bin/env python3
"""
Experiment #495: 6h Primary + 1d HTF — Simplified Trend + Mean Reversion

Hypothesis: 6h timeframe needs faster HTF (1d not 1w) for responsive entries.
Complex regime switching failed in prior 6h experiments (#483-491 all negative).
Simple HMA trend + RSI extremes + Donchian breakout should generate 30-60 trades/year.

Strategy logic:
1. 1d HMA(21) = daily trend bias (HTF filter, faster than 1w)
2. 6h RSI(14) extremes = mean reversion entries (35/65 thresholds, loose)
3. 6h Donchian(20) breakout = trend continuation confirmation
4. 6h ATR(14) volatility filter = avoid entries during extreme vol spikes
5. ATR(14)*2.5 stoploss on all positions
6. OR logic for entries (any trigger works, not AND)

Key changes from failed 6h experiments:
- 1d HTF bias (not 1w which is too slow for 6h entries)
- LOOSE RSI thresholds (35/65 not 30/70)
- No complex regime switching (CHOP failed on 6h)
- No volume filters (volume strategies failed on 6h)
- Simple HMA crossover + RSI + Donchian combination

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=20 test
Timeframe: 6h (first proper 6h experiment with correct MTF alignment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_donchian_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_keltner(high, low, close, atr_period=14, atr_mult=2.0):
    """Keltner Channel - ATR-based bands"""
    n = len(close)
    if n < atr_period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, atr_period)
    middle = pd.Series(close).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    keltner_upper, keltner_middle, keltner_lower = calculate_keltner(high, low, close, atr_period=14, atr_mult=2.0)
    
    # HMA crossover signals
    hma_6h_prev = np.roll(hma_6h, 1)
    hma_cross_bull = (close > hma_6h) & (hma_6h_prev <= close) & (~np.isnan(hma_6h_prev))
    hma_cross_bear = (close < hma_6h) & (hma_6h_prev >= close) & (~np.isnan(hma_6h_prev))
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
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
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65 for entries) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === HMA CROSSOVER ===
        hma_bull_cross = hma_cross_bull[i]
        hma_bear_cross = hma_cross_bear[i]
        
        # === KELTNER SQUEEZE ===
        keltner_long = close[i] > keltner_middle[i] if not np.isnan(keltner_middle[i]) else False
        keltner_short = close[i] < keltner_middle[i] if not np.isnan(keltner_middle[i]) else False
        
        # === VOLATILITY FILTER (avoid extreme vol spikes) ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 2.5  # Avoid entering during 2.5x normal vol
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # TREND LONG: 1d bull + (Donchian breakout OR HMA cross OR RSI recovery)
        if htf_bull and vol_normal:
            if donchian_breakout_long and above_sma50:
                desired_signal = SIZE_STRONG
            elif hma_bull_cross and above_sma50:
                desired_signal = SIZE_BASE
            elif rsi_extreme_oversold and rsi_rising and above_sma50:
                # RSI oversold + starting to rise = mean reversion long
                desired_signal = SIZE_BASE
            elif rsi[i] > 50.0 and rsi[i-1] <= 50.0 and above_sma50:
                # RSI crossing above 50 = momentum shift
                desired_signal = SIZE_BASE * 0.8
        
        # TREND SHORT: 1d bear + (Donchian breakdown OR HMA cross OR RSI weakness)
        elif htf_bear and vol_normal:
            if donchian_breakdown_short and below_sma50:
                desired_signal = -SIZE_STRONG
            elif hma_bear_cross and below_sma50:
                desired_signal = -SIZE_BASE
            elif rsi_extreme_overbought and rsi_falling and below_sma50:
                # RSI overbought + starting to fall = mean reversion short
                desired_signal = -SIZE_BASE
            elif rsi[i] < 50.0 and rsi[i-1] >= 50.0 and below_sma50:
                # RSI crossing below 50 = weakness
                desired_signal = -SIZE_BASE * 0.8
        
        # MEAN REVERSION LONG: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_oversold and above_sma50 and rsi_rising:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and below_sma50 and rsi_falling:
                desired_signal = -SIZE_BASE * 0.8
        
        # KELTNER BREAKOUT: Strong momentum signal
        if desired_signal == 0.0 and vol_normal:
            if keltner_long and donchian_breakout_long and rsi[i] > 50.0:
                desired_signal = SIZE_BASE * 0.8
            elif keltner_short and donchian_breakdown_short and rsi[i] < 50.0:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals