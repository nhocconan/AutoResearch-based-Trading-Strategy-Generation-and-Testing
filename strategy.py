#!/usr/bin/env python3
"""
Experiment #509: 15m Primary + 1h/1d HTF — Fast Trend Pullback Strategy

Hypothesis: 15m timeframe needs FAST HTF (1h not 4h) for responsive entries.
Previous 15m experiments (#497, #499, #501, #505) all got Sharpe=0.000 = ZERO TRADES.
Problem: Entry conditions too strict, never all aligned.

Solution: LOOSE OR logic for entries + 1h HTF (faster than 4h) + volume confirmation.
- 1h HMA(21) = trend bias (fast HTF for 15m entries)
- 15m RSI(7) = entry timing (oversold/overbought, loose 35/65 thresholds)
- 15m HMA(8/21) = momentum confirmation (fast HMA cross)
- Volume spike filter = only enter on 1.5x average volume
- Session filter = UTC 00-12 (London/NY overlap, crypto most liquid)
- ATR(14)*2.5 stoploss on all positions

Key changes from failed 15m experiments:
- 1h HTF bias (not 4h which is too slow for 15m entries)
- LOOSE RSI thresholds (35/65 not 30/70)
- OR logic for entries (any trigger works, not AND)
- Volume confirmation (avoid low-liquidity traps)
- Session filter (prefer high-liquidity hours)

Target: Sharpe>0.40, trades>=160 train (40/year), trades>=30 test
Timeframe: 15m (first viable 15m experiment with correct MTF alignment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_1h1d_v1"
timeframe = "15m"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_spike(volume, period=20):
    """Volume spike detection - returns ratio to SMA"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_sma
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d HMA for regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m_fast = calculate_hma(close, period=8)
    hma_15m_slow = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 15m
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_spike(volume, period=20)
    
    # HMA crossover signals
    hma_fast_prev = np.roll(hma_15m_fast, 1)
    hma_slow_prev = np.roll(hma_15m_slow, 1)
    hma_cross_bull = (hma_15m_fast > hma_15m_slow) & (hma_fast_prev <= hma_slow_prev) & (~np.isnan(hma_fast_prev)) & (~np.isnan(hma_slow_prev))
    hma_cross_bear = (hma_15m_fast < hma_15m_slow) & (hma_fast_prev >= hma_slow_prev) & (~np.isnan(hma_fast_prev)) & (~np.isnan(hma_slow_prev))
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_fast[i]) or np.isnan(hma_15m_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 = London/NY overlap) ===
        hour_utc = (open_time[i] // 3600000) % 24
        session_active = (hour_utc >= 0) and (hour_utc < 12)
        
        # === 1d HTF REGIME ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HTF BIAS ===
        htf_bull = close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_1h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m_slow[i]
        hma_bear = close[i] < hma_15m_slow[i]
        
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
        
        # === HMA CROSSOVER ===
        hma_bull_cross = hma_cross_bull[i]
        hma_bear_cross = hma_cross_bear[i]
        
        # === VOLUME SPIKE FILTER ===
        vol_spike = vol_ratio[i] > 1.5  # 1.5x average volume
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 3.0  # Avoid extreme vol spikes
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # TREND LONG: 1h bull + (HMA cross OR RSI recovery OR pullback)
        if htf_bull and vol_normal:
            if hma_bull_cross and vol_spike:
                desired_signal = SIZE_STRONG
            elif rsi_extreme_oversold and rsi_rising and above_sma50:
                # RSI oversold + starting to rise = mean reversion long
                desired_signal = SIZE_BASE
            elif rsi[i] > 45.0 and rsi[i-1] <= 45.0 and above_sma50:
                # RSI crossing above 45 = momentum shift
                desired_signal = SIZE_BASE
            elif hma_bull and close[i] > hma_15m_fast[i] and session_active:
                # Price above both HMA during active session
                desired_signal = SIZE_BASE * 0.8
        
        # TREND SHORT: 1h bear + (HMA cross OR RSI weakness OR pullback)
        elif htf_bear and vol_normal:
            if hma_bear_cross and vol_spike:
                desired_signal = -SIZE_STRONG
            elif rsi_extreme_overbought and rsi_falling and below_sma50:
                # RSI overbought + starting to fall = mean reversion short
                desired_signal = -SIZE_BASE
            elif rsi[i] < 55.0 and rsi[i-1] >= 55.0 and below_sma50:
                # RSI crossing below 55 = weakness
                desired_signal = -SIZE_BASE
            elif hma_bear and close[i] < hma_15m_fast[i] and session_active:
                # Price below both HMA during active session
                desired_signal = -SIZE_BASE * 0.8
        
        # MEAN REVERSION LONG: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal and session_active:
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_oversold and above_sma50 and rsi_rising:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal and session_active:
            if rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and below_sma50 and rsi_falling:
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