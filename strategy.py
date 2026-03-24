#!/usr/bin/env python3
"""
Experiment #913: 5m Primary + 15m/4h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 5m timeframe with 15m/4h HTF bias provides high-frequency entry timing
while maintaining trend direction from higher timeframes. Session filter (08-20 UTC)
reduces noise during low-liquidity hours. RSI pullback entries in HTF trend direction
provide proven edge with sufficient trade frequency.

Key innovations:
1. 4h HMA(21) for primary trend bias — smooth, low-lag trend filter
2. 15m HMA(16) for intermediate trend confirmation
3. 5m RSI(7) for pullback entries — faster than standard RSI(14)
4. Session filter: 08:00-20:00 UTC only (avoid Asian night low-liquidity)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller due to 5m frequency)

Entry conditions (LOOSE to ensure ≥50 trades/train, ≥5/test):
- LONG: 4h HMA bull + 15m HMA bull + 5m RSI(7) < 35 + session active
- SHORT: 4h HMA bear + 15m HMA bear + 5m RSI(7) > 65 + session active

Target: Sharpe>0.45, trades>=50 train, trades>=5 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_pullback_session_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
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

def get_session_active(prices):
    """
    Session filter: 08:00-20:00 UTC only
    Returns boolean array indicating active trading hours
    """
    n = len(prices)
    session_active = np.zeros(n, dtype=bool)
    
    # Parse open_time to get hour
    for i in range(n):
        open_time = prices.iloc[i]['open_time']
        # open_time is in milliseconds since epoch
        hour = (open_time // (1000 * 60 * 60)) % 24
        # Active session: 08:00-20:00 UTC (12 hours of highest liquidity)
        if 8 <= hour < 20:
            session_active[i] = True
    
    return session_active

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=16)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate 5m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    session_active = get_session_active(prices)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER ===
        if not session_active[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === INTERMEDIATE TREND (15m HMA) ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === RSI PULLBACK CONDITIONS (LOOSE for trade frequency) ===
        rsi_oversold = rsi_7[i] < 35.0  # Pullback long
        rsi_overbought = rsi_7[i] > 65.0  # Pullback short
        rsi_neutral_long = rsi_7[i] < 45.0  # Looser long
        rsi_neutral_short = rsi_7[i] > 55.0  # Looser short
        
        # === ENTRY LOGIC (HTF TREND + RSI PULLBACK) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m bull + RSI pullback
        if htf_4h_bull and htf_15m_bull:
            if rsi_oversold:
                desired_signal = SIZE_STRONG
            elif rsi_neutral_long:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m bear + RSI pullback
        elif htf_4h_bear and htf_15m_bear:
            if rsi_overbought:
                desired_signal = -SIZE_STRONG
            elif rsi_neutral_short:
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
                entry_atr = atr_14[i]
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