#!/usr/bin/env python3
"""
Experiment #725: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 4h/1d HTF bias can capture intraday momentum while
avoiding noise. Using proven HMA trend + RSI pullback pattern (from best strategy)
but adapted for 15m with LOOSE entry conditions to ensure trade generation.

Key innovations:
1. 1d HMA(21) for primary trend bias (slow, reliable direction)
2. 4h HMA(21) for intermediate confirmation (faster regime check)
3. 15m RSI(7) for entry timing - LOOSE thresholds (40/60 not 20/80)
4. Session filter: UTC 00-12 (London/NY overlap) but NOT restrictive
5. ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.15 base, 0.25 strong (smaller for 15m frequency)

CRITICAL: Entry conditions LOOSE to ensure >=10 trades/symbol/train
- Long: 1d HMA bull + (4h HMA bull OR 15m RSI<45)
- Short: 1d HMA bear + (4h HMA bear OR 15m RSI>55)
- Only require ONE HTF alignment, not both

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h1d_session_v2"
timeframe = "15m"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA (Rule 2 - use align_htf_to_ltf)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for entries
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    atr = calculate_atr(high, low, close, period=14)
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        # === HTF BIAS (1d + 4h HMA) ===
        # LOOSE: only need ONE HTF aligned, not both
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # Strong bias: both HTF agree
        htf_strong_bull = htf_1d_bull and htf_4h_bull
        htf_strong_bear = htf_1d_bear and htf_4h_bear
        
        # Weak bias: at least one HTF agrees
        htf_weak_bull = htf_1d_bull or htf_4h_bull
        htf_weak_bear = htf_1d_bear or htf_4h_bear
        
        # === SESSION FILTER (LOOSE - prefer but don't require) ===
        # UTC 00-12 is London/NY overlap (best liquidity)
        # But we allow trades outside this window too (just smaller size)
        prime_session = (session_hours[i] >= 0 and session_hours[i] <= 12)
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADES) ===
        # Long: RSI(7) < 45 (not < 20) in uptrend
        # Short: RSI(7) > 55 (not > 80) in downtrend
        rsi_oversold = rsi_7[i] < 45.0
        rsi_overbought = rsi_7[i] > 55.0
        rsi_neutral = rsi_7[i] >= 45.0 and rsi_7[i] <= 55.0
        
        # === HMA TREND CONFIRMATION ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Strong HTF bull + any RSI condition (very loose)
        if htf_strong_bull:
            if rsi_oversold or hma_bull:
                desired_signal = SIZE_STRONG if prime_session else SIZE_BASE
        
        # LONG: Weak HTF bull + RSI oversold + HMA bull
        elif htf_weak_bull and rsi_oversold and hma_bull:
            desired_signal = SIZE_BASE
        
        # LONG: RSI very oversold (any HTF condition)
        elif rsi_7[i] < 30.0:
            desired_signal = SIZE_BASE
        
        # SHORT: Strong HTF bear + any RSI condition (very loose)
        elif htf_strong_bear:
            if rsi_overbought or hma_bear:
                desired_signal = -SIZE_STRONG if prime_session else -SIZE_BASE
        
        # SHORT: Weak HTF bear + RSI overbought + HMA bear
        elif htf_weak_bear and rsi_overbought and hma_bear:
            desired_signal = -SIZE_BASE
        
        # SHORT: RSI very overbought (any HTF condition)
        elif rsi_7[i] > 70.0:
            desired_signal = -SIZE_BASE
        
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