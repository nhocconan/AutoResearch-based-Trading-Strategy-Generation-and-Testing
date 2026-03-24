#!/usr/bin/env python3
"""
Experiment #633: 5m Primary + 15m/4h HTF — Session + Trend + Pullback

Hypothesis: 5m timeframe is unexplored territory. Success requires:
1. 4h HMA for macro trend bias (only trade WITH trend, never counter)
2. 15m RSI for momentum filter (avoid entering at extremes)
3. Session filter 08-20 UTC (high liquidity, avoid Asia overnight)
4. ATR volatility regime (only trade when vol is normal, not spiking)
5. 5m EMA pullback for entry timing precision

Key differences from failed lower TF strategies:
- Session filter MANDATORY (most 15m failures lacked this)
- HTF trend is HARD FILTER (not optional boost)
- Smaller size (0.15-0.20) due to higher trade frequency
- Volatility filter blocks entries during panic/spike conditions

Entry logic:
- LONG: 4h HMA bull + 15m RSI 35-65 + session active + vol normal + price>5m EMA
- SHORT: 4h HMA bear + 15m RSI 35-65 + session active + vol normal + price<5m EMA

Target: 50-120 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 5m
Size: 0.15 base, 0.20 strong confluence
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_trend_pullback_15m4h_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def get_session_active(open_time):
    """
    Session filter: 08-20 UTC (high liquidity hours)
    open_time is in milliseconds since epoch
    Returns 1 if active, 0 if inactive
    """
    # Convert to hour of day UTC
    hour = (open_time // 3600000) % 24
    # Active during 08-20 UTC (12 hours of high liquidity)
    return 1 if 8 <= hour < 20 else 0

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
    ema_5m = calculate_ema(close, period=21)
    atr_5m = calculate_atr(high, low, close, period=14)
    atr_5m_long = calculate_atr(high, low, close, period=50)
    
    # Session filter
    session_active = np.array([get_session_active(ot) for ot in open_time])
    
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
        if np.isnan(atr_5m[i]) or atr_5m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_5m[i]) or np.isnan(rsi_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Volatility regime: ATR(14)/ATR(50) should be 0.7-1.5 (normal vol)
        if np.isnan(atr_5m_long[i]) or atr_5m_long[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        vol_ratio = atr_5m[i] / atr_5m_long[i]
        vol_normal = 0.7 <= vol_ratio <= 1.8  # Allow some spike tolerance
        
        # Session filter
        session_ok = session_active[i] == 1
        
        # === HTF TREND BIAS (4h HMA - HARD FILTER) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI MOMENTUM FILTER (avoid extremes) ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_ok_long = 35.0 <= rsi_15m <= 70.0  # Not overbought
        rsi_ok_short = 30.0 <= rsi_15m <= 65.0  # Not oversold
        
        # RSI momentum (improving for long, weakening for short)
        rsi_momentum_long = False
        rsi_momentum_short = False
        if i > 1 and not np.isnan(rsi_15m_aligned[i-1]):
            rsi_momentum_long = rsi_15m > rsi_15m_aligned[i-1]
            rsi_momentum_short = rsi_15m < rsi_15m_aligned[i-1]
        
        # === 5m PULLBACK ENTRY ===
        # Long: price above EMA but pulling back (close > EMA, RSI not extreme)
        # Short: price below EMA but bouncing (close < EMA, RSI not extreme)
        price_above_ema = close[i] > ema_5m[i]
        price_below_ema = close[i] < ema_5m[i]
        
        # Distance from EMA (pullback depth)
        ema_distance = (close[i] - ema_5m[i]) / ema_5m[i] * 100 if ema_5m[i] > 1e-10 else 0
        pullback_ok_long = -3.0 <= ema_distance <= 2.0  # Small pullback in uptrend
        pullback_ok_short = -2.0 <= ema_distance <= 3.0  # Small bounce in downtrend
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: All filters must align
        if htf_bull and rsi_ok_long and session_ok and vol_normal and price_above_ema and pullback_ok_long:
            if rsi_momentum_long:
                desired_signal = SIZE_STRONG  # Strong confluence
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: All filters must align
        elif htf_bear and rsi_ok_short and session_ok and vol_normal and price_below_ema and pullback_ok_short:
            if rsi_momentum_short:
                desired_signal = -SIZE_STRONG  # Strong confluence
            else:
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr_5m[i]
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