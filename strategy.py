#!/usr/bin/env python3
"""
Experiment #881: 15m Primary + 1h/4h HTF — HMA Trend + RSI Momentum + Session Filter

Hypothesis: 15m timeframe with 4h trend bias and 1h momentum confirmation provides
optimal balance between trade frequency (40-100/year) and signal quality. Previous
15m experiments failed due to overly strict entry conditions (0 trades generated).

Key innovations for 15m:
1. 4h HMA(21) for HTF trend bias - smoother trend filter
2. 1h RSI(14) for momentum confirmation - loose thresholds (35/65 not 30/70)
3. 15m EMA(21) for entry timing - fast response to price action
4. Session filter: 00-12 UTC preferred but not mandatory
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.15-0.20 (smaller for 15m frequency)

CRITICAL: Entry conditions LOOSE to ensure ≥10 trades/train, ≥3/test
- RSI > 35 for longs (not > 50)
- RSI < 65 for shorts (not < 50)
- Session preference, not requirement
- 4h trend + 1h momentum + 15m entry = 3 confluence

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_4h1h_v1"
timeframe = "15m"
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
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    ema_15m_21 = calculate_ema(close, period=21)
    ema_15m_50 = calculate_ema(close, period=50)
    rsi_15m_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(ema_15m_21[i]) or np.isnan(ema_15m_50[i]) or np.isnan(rsi_15m_14[i]):
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
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        hour = get_session_hour(open_time[i])
        session_active = (hour >= 0 and hour <= 12)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (RSI) - LOOSE THRESHOLDS ===
        rsi_1h = rsi_1h_aligned[i]
        momentum_bull = rsi_1h > 35.0  # Not oversold
        momentum_bear = rsi_1h < 65.0  # Not overbought
        momentum_strong_bull = rsi_1h > 45.0
        momentum_strong_bear = rsi_1h < 55.0
        
        # === 15m ENTRY SIGNALS ===
        ema_15m_bull = ema_15m_21[i] > ema_15m_50[i]
        ema_15m_bear = ema_15m_21[i] < ema_15m_50[i]
        
        rsi_15m = rsi_15m_14[i]
        rsi_15m_bull = rsi_15m > 40.0  # Loose long
        rsi_15m_bear = rsi_15m < 60.0  # Loose short
        
        price_above_ema = close[i] > ema_15m_21[i]
        price_below_ema = close[i] < ema_15m_21[i]
        
        # === ENTRY LOGIC (3+ CONFLUENCE, LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG conditions: 4h bull + 1h momentum + 15m entry
        if htf_4h_bull and momentum_bull:
            confluence_count = 0
            if ema_15m_bull:
                confluence_count += 1
            if rsi_15m_bull:
                confluence_count += 1
            if price_above_ema:
                confluence_count += 1
            if session_active:
                confluence_count += 0.5  # Session is bonus
            
            # Need 2+ confluence for entry
            if confluence_count >= 2.0:
                if momentum_strong_bull and confluence_count >= 2.5:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT conditions: 4h bear + 1h momentum + 15m entry
        elif htf_4h_bear and momentum_bear:
            confluence_count = 0
            if ema_15m_bear:
                confluence_count += 1
            if rsi_15m_bear:
                confluence_count += 1
            if price_below_ema:
                confluence_count += 1
            if session_active:
                confluence_count += 0.5  # Session is bonus
            
            # Need 2+ confluence for entry
            if confluence_count >= 2.0:
                if momentum_strong_bear and confluence_count >= 2.5:
                    desired_signal = -SIZE_STRONG
                else:
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