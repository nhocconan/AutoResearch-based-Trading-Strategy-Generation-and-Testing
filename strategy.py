#!/usr/bin/env python3
"""
Experiment #653: 5m Primary + 15m/4h HTF — Session-Filtered Pullback Strategy

Hypothesis: 5m timeframe has ZERO prior experiments. Key insight: 5m breakouts whipsaw badly,
but pullback entries WITHIN established HTF trend work better. Use 4h HMA for primary bias,
15m RSI for pullback detection, 5m for entry timing. CRITICAL: Session filter (13-17 UTC
London/NY overlap) to avoid low-liquidity chop. This should reduce false signals while
maintaining trade frequency.

Key innovations:
1. 4h HMA(21) - primary trend bias (long only above, short only below)
2. 15m RSI(14) - pullback detection (RSI<40 in uptrend, RSI>60 in downtrend)
3. 5m session filter - only trade 13-17 UTC (highest liquidity, lowest slippage)
4. 5m EMA(8/21) crossover - entry timing precision within HTF trend
5. ATR(14) trailing stop - 2.5x for risk management
6. Discrete sizing: 0.15 base, 0.20 strong (small size for 5m fee drag)

Entry conditions (LOOSE to ensure trades on all symbols):
- LONG: 4h HMA bull + 15m RSI < 50 (pullback) + 5m EMA8 > EMA21 + session active
- SHORT: 4h HMA bear + 15m RSI > 50 (pullback) + 5m EMA8 < EMA21 + session active
- Exit: EMA cross against position OR stoploss hit

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 5m
Size: 0.15-0.20 discrete (small for 5m fee management)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_pullback_hma_rsi_ema_v1"
timeframe = "5m"
leverage = 1.0

def calculate_ema(close, span):
    """Exponential Moving Average"""
    n = len(close)
    if n < span:
        return np.full(n, np.nan)
    ema = pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values
    return ema

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
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

def calculate_hma(close, period):
    """Hull Moving Average for HTF"""
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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align HTF HMA (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF RSI (15m)
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
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
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
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
        
        # === SESSION FILTER (13-17 UTC London/NY overlap) ===
        session_active = (utc_hour[i] >= 13) and (utc_hour[i] <= 17)
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI PULLBACK ===
        rsi_neutral = (rsi_15m_aligned[i] >= 35.0) and (rsi_15m_aligned[i] <= 65.0)
        rsi_pullback_long = rsi_15m_aligned[i] < 50.0  # Pullback in uptrend
        rsi_pullback_short = rsi_15m_aligned[i] > 50.0  # Pullback in downtrend
        
        # === 5m EMA CROSSOVER ===
        ema_bull = ema_8[i] > ema_21[i]
        ema_bear = ema_8[i] < ema_21[i]
        
        # Check EMA cross direction (momentum confirmation)
        ema_cross_long = False
        ema_cross_short = False
        if i >= 2 and not np.isnan(ema_8[i-1]) and not np.isnan(ema_21[i-1]):
            ema_cross_long = (ema_8[i-1] <= ema_21[i-1]) and (ema_8[i] > ema_21[i])
            ema_cross_short = (ema_8[i-1] >= ema_21[i-1]) and (ema_8[i] < ema_21[i])
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI pullback + 5m EMA bull + session active
        if session_active and htf_bull and rsi_pullback_long and ema_bull:
            if ema_cross_long:
                desired_signal = SIZE_STRONG  # Fresh cross = stronger signal
            else:
                desired_signal = SIZE_BASE  # Already in trend
        
        # SHORT: 4h bear + 15m RSI pullback + 5m EMA bear + session active
        elif session_active and htf_bear and rsi_pullback_short and ema_bear:
            if ema_cross_short:
                desired_signal = -SIZE_STRONG  # Fresh cross = stronger signal
            else:
                desired_signal = -SIZE_BASE  # Already in trend
        
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
        
        # === EXIT ON EMA CROSS AGAINST POSITION ===
        if in_position and position_side > 0 and ema_bear:
            desired_signal = 0.0  # Exit long on EMA bear cross
        
        if in_position and position_side < 0 and ema_bull:
            desired_signal = 0.0  # Exit short on EMA bull cross
        
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