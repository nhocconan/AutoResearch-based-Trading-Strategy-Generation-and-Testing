#!/usr/bin/env python3
"""
Experiment #481: 15m Primary + 1h/4h/1d HTF — Loose Confluence for Trade Generation

Hypothesis: Previous 15m experiments (#469, #477) failed with 0 trades due to overly
strict entry conditions. This strategy uses LOOSE filters with OR logic to guarantee
trade generation while maintaining quality via HTF trend bias.

Key design:
1. 4h HMA(21) = primary trend bias (proven in #478)
2. 1h RSI(14) = momentum confirmation (loose: 40/60 not 30/70)
3. 15m triggers: Donchian(20) breakout OR RSI(7) extreme OR session momentum
4. Session filter: 00-12 UTC preferred (London/NY overlap)
5. ATR(14)*2.5 stoploss on all positions
6. OR logic for entries (any 2 of 3 conditions = entry)

Target: Sharpe>0.40, trades>=100 train, trades>=15 test, DD>-40%
Timeframe: 15m (first proper 15m experiment)
Position size: 0.20-0.25 (conservative for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_loose_confluence_hma_rsi_donchian_4h1h_v1"
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // 3600000) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1h RSI for momentum
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_15m_std = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(rsi_15m_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
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
        
        # === 4h HTF BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (LOOSE: 40/60) ===
        mom_bull = rsi_1h_aligned[i] > 40.0
        mom_bear = rsi_1h_aligned[i] < 60.0
        mom_strong_bull = rsi_1h_aligned[i] > 50.0
        mom_strong_bear = rsi_1h_aligned[i] < 50.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === 15m RSI EXTREMES (LOOSE for 15m) ===
        rsi_os = rsi_15m[i] < 35.0
        rsi_ob = rsi_15m[i] > 65.0
        rsi_extreme_os = rsi_15m[i] < 25.0
        rsi_extreme_ob = rsi_15m[i] > 75.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        hour = get_session_hour(open_time[i])
        prime_session = 0 <= hour < 12  # London/NY overlap
        
        # === ENTRY LOGIC (LOOSE - OR logic, need 2 of 3) ===
        desired_signal = 0.0
        
        # Count confluence for LONG
        long_confluence = 0
        if htf_bull:
            long_confluence += 1
        if mom_bull:
            long_confluence += 1
        if hma_bull:
            long_confluence += 1
        
        # Count confluence for SHORT
        short_confluence = 0
        if htf_bear:
            short_confluence += 1
        if mom_bear:
            short_confluence += 1
        if hma_bear:
            short_confluence += 1
        
        # TREND LONG: 4h bull + 1h mom + (Donchian OR RSI recovery OR HMA cross)
        if long_confluence >= 2:
            if donchian_breakout_long:
                desired_signal = SIZE_STRONG if prime_session else SIZE_BASE
            elif rsi_15m[i] > 40.0 and rsi_15m[i-1] < 40.0:
                # RSI crossing above 40 = momentum shift
                desired_signal = SIZE_BASE
            elif hma_bull and above_sma50:
                desired_signal = SIZE_BASE
        
        # TREND SHORT: 4h bear + 1h mom + (Donchian OR RSI weakness OR HMA cross)
        elif short_confluence >= 2:
            if donchian_breakdown_short:
                desired_signal = -SIZE_STRONG if prime_session else -SIZE_BASE
            elif rsi_15m[i] < 60.0 and rsi_15m[i-1] > 60.0:
                # RSI crossing below 60 = weakness
                desired_signal = -SIZE_BASE
            elif hma_bear and below_sma50:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: RSI extreme + SMA200 support (no HTF filter)
        if desired_signal == 0.0:
            if rsi_extreme_os and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_os and above_sma50 and htf_bull:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme + SMA200 resistance (no HTF filter)
        if desired_signal == 0.0:
            if rsi_extreme_ob and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_ob and below_sma50 and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
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