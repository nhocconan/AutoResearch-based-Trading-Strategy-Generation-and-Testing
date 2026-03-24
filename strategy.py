#!/usr/bin/env python3
"""
Experiment #373: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m is extremely noisy and has ZERO prior experiments. Success requires:
1. HEAVY HTF filtering (4h for trend direction ONLY)
2. 15m for pullback detection (not 5m RSI which is too noisy)
3. 5m ONLY for precise entry timing within session hours
4. Session filter 08-20 UTC (highest liquidity, lowest slippage)
5. Small position size (0.15-0.20) due to higher trade frequency

Key differences from failed 15m strategies:
- Even stricter HTF alignment (4h trend MUST agree)
- Session filter MANDATORY (08-20 UTC only)
- Smaller size (0.15 base) to handle fee drag
- Only trade WITH 4h trend (no counter-trend on 5m)

Entry Logic:
- Long: 4h HMA bull + 15m RSI < 40 (pullback) + 5m close > 5m EMA(9) + session hours
- Short: 4h HMA bear + 15m RSI > 60 (pullback) + 5m close < 5m EMA(9) + session hours

Target: Sharpe>0.40, DD>-35%, trades>=50 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_pullback_15m4h_v1"
timeframe = "5m"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def get_session_hour(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # Convert milliseconds to seconds, then to datetime
    ts_seconds = open_time / 1000.0
    # Get hour in UTC
    hour = (ts_seconds % 86400) // 3600
    return int(hour)

def is_session_active(open_time):
    """Check if bar is within 08-20 UTC session"""
    hour = get_session_hour(open_time)
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for pullback detection
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    ema_5m_fast = calculate_ema(close, 9)
    ema_5m_slow = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(ema_5m_fast[i]) or np.isnan(ema_5m_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC ONLY) ===
        in_session = is_session_active(open_time[i])
        
        # === 4h TREND BIAS (PRIMARY DIRECTION) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI PULLBACK DETECTION ===
        # For long: RSI pulled back to 35-45 range (not oversold, just pullback)
        # For short: RSI pulled back to 55-65 range (not overbought, just pullback)
        rsi_15m_pullback_long = 30.0 < rsi_15m_aligned[i] < 50.0
        rsi_15m_pullback_short = 50.0 < rsi_15m_aligned[i] < 70.0
        
        # === 5m MOMENTUM CONFIRMATION ===
        # Fast EMA above slow EMA for long momentum
        ema_bull = ema_5m_fast[i] > ema_5m_slow[i]
        ema_bear = ema_5m_fast[i] < ema_5m_slow[i]
        
        # Price above fast EMA for additional confirmation
        price_above_ema = close[i] > ema_5m_fast[i]
        price_below_ema = close[i] < ema_5m_fast[i]
        
        # === 5m RSI CONFIRMATION (not extreme, just direction) ===
        rsi_5m_bull = 40.0 < rsi_5m[i] < 70.0
        rsi_5m_bear = 30.0 < rsi_5m[i] < 60.0
        
        # === ENTRY LOGIC (STRICT - ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m pullback + 5m momentum + session
        if htf_4h_bull and rsi_15m_pullback_long and ema_bull and price_above_ema and in_session:
            # Additional confirmation: 5m RSI not overbought
            if rsi_5m_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m pullback + 5m momentum + session
        elif htf_4h_bear and rsi_15m_pullback_short and ema_bear and price_below_ema and in_session:
            # Additional confirmation: 5m RSI not oversold
            if rsi_5m_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
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