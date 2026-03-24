#!/usr/bin/env python3
"""
Experiment #597: 15m Primary + 4h/1d HTF — Simple RSI Mean Reversion with HTF Bias

Hypothesis: Previous 15m experiments (#585, #589, #593, #596) ALL failed with Sharpe=0.000
because entry conditions were TOO STRICT (0 trades generated). This strategy SIMPLIFIES
entry logic to ensure trades occur while using HTF for direction bias only.

Key changes from failed 15m attempts:
1. SIMPLER entry: RSI(7) extremes only (no 5+ confluence requirements)
2. HTF = direction filter ONLY (4h HMA tells long/short bias, not entry trigger)
3. Session filter relaxed: 00-14 UTC (covers London+NY overlap)
4. Looser RSI thresholds: <30/>70 instead of <20/>80
5. Position sizing: 0.15 base, 0.25 with HTF alignment (smaller for 15m frequency)
6. ATR stoploss: 2.0x (tighter than 2.5x for faster turnover)

Strategy logic:
1. 4h HMA(21) = trend bias (long only above, short only below)
2. 1d HMA(21) = macro confirmation (optional boost to position size)
3. 15m RSI(7) = entry trigger (oversold <30 long, overbought >70 short)
4. Session filter = 00-14 UTC only (liquid hours)
5. ATR(14)*2.0 = stoploss on all positions

Target: 40-100 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_simple_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index - Wilder's method"""
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
    """Average True Range - Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-14 UTC for liquid hours) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc <= 14)
        
        # === HTF BIAS (4h HMA = trend direction) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for macro confirmation (boosts position size)
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI EXTREMES (fast RSI(7) for 15m) ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # RSI recovery confirmation (reduces false signals)
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLIFIED for trade generation) ===
        desired_signal = 0.0
        
        # LONG entries: RSI oversold + in session + HTF bias helps
        if in_session and rsi_oversold:
            if htf_bull:
                # HTF bullish = strong long signal
                if macro_bull:
                    desired_signal = SIZE_STRONG  # 0.25
                else:
                    desired_signal = SIZE_BASE  # 0.15
            else:
                # HTF bearish but RSI extreme = mean reversion bounce
                if rsi_rising:
                    desired_signal = SIZE_BASE * 0.8  # 0.12
        
        # SHORT entries: RSI overbought + in session + HTF bias helps
        elif in_session and rsi_overbought:
            if htf_bear:
                # HTF bearish = strong short signal
                if macro_bear:
                    desired_signal = -SIZE_STRONG  # -0.25
                else:
                    desired_signal = -SIZE_BASE  # -0.15
            else:
                # HTF bullish but RSI extreme = mean reversion drop
                if rsi_falling:
                    desired_signal = -SIZE_BASE * 0.8  # -0.12
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop: highest - 2.0*ATR
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop: lowest + 2.0*ATR
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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