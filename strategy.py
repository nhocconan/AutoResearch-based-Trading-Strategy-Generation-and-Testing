#!/usr/bin/env python3
"""
Experiment #997: 15m Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 4h HMA trend bias + 15m RSI pullback entries will capture
intraday swings while respecting higher-timeframe direction. Session filter (UTC 00-12)
reduces low-liquidity noise. LOOSE entry conditions ensure sufficient trade generation.

Key innovations:
1. 4h HMA(21) for intermediate trend bias (not too strict)
2. 15m RSI(7) for entry timing (thresholds 35/65, not extreme 20/80)
3. 12h momentum filter (close > open) for weekly bias confirmation
4. Session filter: UTC 00-12 (London + NY overlap for crypto liquidity)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete signal sizes: 0.20 base, 0.25 strong

Why 15m can work:
- Captures intraday mean reversion within HTF trend
- More entries than 1h/4h but filtered by session + HTF
- RSI(7) is responsive enough for 15m without excessive noise

Entry conditions (LOOSE to guarantee trades):
- LONG = 4h bull + 12h bull + RSI(7)<40 + session 00-12 UTC
- SHORT = 4h bear + 12h bear + RSI(7)>60 + session 00-12 UTC
- Relaxed RSI thresholds (40/60 instead of 30/70) for more trades

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_prices = prices["open"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # 12h momentum: close vs open (weekly bias proxy)
    momentum_12h_raw = (df_12h['close'].values - df_12h['open'].values) / (df_12h['open'].values + 1e-10)
    momentum_12h_aligned = align_htf_to_ltf(prices, df_12h, momentum_12h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Get open_time for session filter
    open_times = prices["open_time"].values
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(momentum_12h_aligned[i]):
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
        
        # === SESSION FILTER (UTC 00-12) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_times[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === HTF BIAS (4h HMA + 12h momentum) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = momentum_12h_aligned[i] > 0.0
        htf_12h_bear = momentum_12h_aligned[i] < 0.0
        
        # === RSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_7[i] < 40  # Relaxed from 30
        rsi_overbought = rsi_7[i] > 60  # Relaxed from 70
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (need HTF bull + RSI oversold + session)
        if htf_4h_bull and htf_12h_bull:
            if rsi_oversold and in_session:
                desired_signal = SIZE_STRONG
            elif rsi_7[i] < 50 and in_session:
                # Even looser: any RSI < 50 in bull regime during session
                desired_signal = SIZE_BASE
        
        # SHORT entries (need HTF bear + RSI overbought + session)
        elif htf_4h_bear and htf_12h_bear:
            if rsi_overbought and in_session:
                desired_signal = -SIZE_STRONG
            elif rsi_7[i] > 50 and in_session:
                # Even looser: any RSI > 50 in bear regime during session
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