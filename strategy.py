#!/usr/bin/env python3
"""
Experiment #629: 15m Primary + 1h/1d HTF — RSI Mean Reversion with HTF Trend Bias

Hypothesis: 15m timeframe can work IF we use HTF for direction and 15m only for entry timing.
Previous 15m experiments failed with 0 trades due to overly strict conditions.

Key changes from failed 15m strategies:
1. LOOSE RSI zones (25/75 instead of 30/70) to ensure trades generate
2. HTF bias only MODIFIES size, doesn't block entries (critical for trade count)
3. Faster RSI(7) for 15m entries (more responsive than RSI(14))
4. Session filter: 00-12 UTC only (London/NY overlap = better fills)
5. Size: 0.15-0.20 (smaller for 15m frequency to reduce fee drag)
6. ATR stoploss: 2.5x to avoid premature exits

Strategy logic:
1. 1d HMA(21) = macro bias (bull/bear, modifies confidence not blocks)
2. 1h HMA(21) = medium trend (same - confidence modifier)
3. 15m RSI(7) = entry trigger (fast for 15m)
4. 15m ATR(14) = stoploss tracking
5. Session: hours 00-12 UTC only (reduces trades to target 40-100/yr)

Entry conditions (LOOSE to ensure trades):
- LONG: RSI(7) < 25 (oversold) + any HTF bias (size varies)
- SHORT: RSI(7) > 75 (overbought) + any HTF bias (size varies)
- HTF alignment boosts size, misalignment reduces but doesn't block

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_htf_bias_1h1d_session_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 15m"""
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
    """Average True Range for stoploss"""
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing for 15m (smaller due to higher frequency)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: 00-12 UTC only (London/NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === HTF BIAS (confidence modifier, NOT hard filter) ===
        # 1d macro bias
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1h medium bias
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # Count HTF alignments
        htf_bull_count = int(htf_1d_bull) + int(htf_1h_bull)
        htf_bear_count = int(htf_1d_bear) + int(htf_1h_bear)
        
        # === RSI ENTRY ZONES (LOOSE for trade generation) ===
        rsi_oversold = rsi[i] < 25.0  # Long entry
        rsi_overbought = rsi[i] > 75.0  # Short entry
        
        # RSI momentum confirmation
        rsi_rising = (i > 0) and (not np.isnan(rsi[i-1])) and (rsi[i] > rsi[i-1])
        rsi_falling = (i > 0) and (not np.isnan(rsi[i-1])) and (rsi[i] < rsi[i-1])
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: RSI oversold (primary trigger)
        if rsi_oversold:
            # Strong: HTF also bull + RSI rising + in session
            if htf_bull_count >= 1 and rsi_rising and in_session:
                if htf_bull_count >= 2:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Base: RSI oversold only (ensure trades generate)
            elif in_session:
                desired_signal = SIZE_BASE * 0.75
        
        # SHORT: RSI overbought (primary trigger)
        elif rsi_overbought:
            # Strong: HTF also bear + RSI falling + in session
            if htf_bear_count >= 1 and rsi_falling and in_session:
                if htf_bear_count >= 2:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Base: RSI overbought only (ensure trades generate)
            elif in_session:
                desired_signal = -SIZE_BASE * 0.75
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE
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