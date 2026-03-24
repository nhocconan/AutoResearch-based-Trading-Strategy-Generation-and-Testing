#!/usr/bin/env python3
"""
Experiment #713: 5m Primary + 15m/4h HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: 5m timeframe is unexplored territory. Using 4h HMA for primary trend bias,
15m RSI for momentum confirmation, and 5m Fisher Transform for precise entry timing.
Session filter (08-20 UTC) ensures liquidity and reduces noise during Asian session.
Fisher Transform excels at catching reversals in bear/range markets (2025 test period).

Key innovations:
1. 4h HMA(21) - primary trend direction (aligned properly via mtf_data)
2. 15m RSI(14) - momentum confirmation (not extreme values, just direction)
3. 5m Fisher Transform(9) - entry timing, crosses at -1.5/+1.5 levels
4. Session filter 08-20 UTC - avoid Asian session noise
5. ATR(14) 2.5x trailing stop - risk management
6. Discrete sizing: 0.0, ±0.15, ±0.25 (smaller for 5m due to fee drag)
7. Asymmetric bias: prefer shorts in bear regime (price < 4h HMA)

Entry conditions (balanced for trade generation):
- LONG: 4h HMA bull + 15m RSI > 45 + Fisher crosses above -1.5 + session active
- SHORT: 4h HMA bear + 15m RSI < 55 + Fisher crosses below +1.5 + session active

Target: Sharpe>0.40, trades>=50 train, trades>=5 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.25 discrete (smaller due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_hma_rsi_session_15m4h_v1"
timeframe = "5m"
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

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals well in bear/range markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to -1 to +1 range
        value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * (fisher[i-1] if i > period else 0)
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if timestamp is within active session (UTC)
    08-20 UTC covers London open through NY session
    """
    # open_time is in milliseconds
    hour = (open_time // 3600000) % 24
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    fisher, trigger = calculate_fisher(close, period=9)
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
    
    # Track Fisher crosses to avoid repeated signals
    prev_fisher = np.nan
    prev_trigger = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI MOMENTUM ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_long_bias = rsi_15m > 45.0  # Not oversold, momentum building
        rsi_short_bias = rsi_15m < 55.0  # Not overbought, momentum weakening
        
        # === FISHER TRANSFORM ENTRY ===
        fisher_cross_long = (prev_fisher <= -1.5 and fisher[i] > -1.5) if not np.isnan(prev_fisher) else False
        fisher_cross_short = (prev_fisher >= 1.5 and fisher[i] < 1.5) if not np.isnan(prev_fisher) else False
        
        # Also allow entries when Fisher is at extremes (mean reversion)
        fisher_extreme_long = fisher[i] < -1.8 and fisher[i] > trigger[i]
        fisher_extreme_short = fisher[i] > 1.8 and fisher[i] < trigger[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI ok + Fisher signal + session active
        if session_active and htf_bull and rsi_long_bias:
            if fisher_cross_long or fisher_extreme_long:
                desired_signal = SIZE_STRONG
            elif fisher[i] > -1.0 and fisher[i] > trigger[i]:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m RSI ok + Fisher signal + session active
        elif session_active and htf_bear and rsi_short_bias:
            if fisher_cross_short or fisher_extreme_short:
                desired_signal = -SIZE_STRONG
            elif fisher[i] < 1.0 and fisher[i] < trigger[i]:
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
        
        # Update previous Fisher values for next iteration
        prev_fisher = fisher[i]
        prev_trigger = trigger[i]
    
    return signals