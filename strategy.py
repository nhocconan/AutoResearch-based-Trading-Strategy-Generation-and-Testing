#!/usr/bin/env python3
"""
Experiment #869: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 1h/1d HTF bias captures intraday moves while avoiding
noise of pure 15m strategies. Key insight from 700+ failures: 15m strategies fail
because entry conditions are TOO STRICT (0 trades). This strategy uses LOOSE entries
with HTF confirmation to ensure ≥10 trades/train, ≥3/test.

Strategy components:
1. 1d HMA(21) = major trend bias (bull/bear regime)
2. 1h HMA(16/48) = intermediate trend confirmation
3. 15m RSI(7) = entry timing on pullbacks (oversold long, overbought short)
4. Session filter = 00-12 UTC (London/NY overlap, higher volume)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.25 (smaller size for 15m frequency)

Entry logic (LOOSE to guarantee trades):
- LONG: 1d HMA bull + 1h HMA bull + RSI(7) < 40 (or <30 for strong)
- SHORT: 1d HMA bear + 1h HMA bear + RSI(7) > 60 (or >70 for strong)
- Session: prefer 00-12 UTC but allow 24h for crypto

Target: Sharpe>0.45, trades>=40/train, trades>=5/test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller than 6h/12h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_loose_v1"
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
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to seconds, then to datetime, extract hour
    seconds = open_time / 1000.0
    # Use numpy datetime64 for efficiency
    hours = ((seconds % 86400) / 3600).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1h_16_raw = calculate_hma(df_1h['close'].values, period=16)
    hma_1h_48_raw = calculate_hma(df_1h['close'].values, period=48)
    hma_1h_16_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_16_raw)
    hma_1h_48_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_48_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1h_16_aligned[i]) or np.isnan(hma_1h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HMA TREND ===
        htf_1h_bull = hma_1h_16_aligned[i] > hma_1h_48_aligned[i]
        htf_1h_bear = hma_1h_16_aligned[i] < hma_1h_48_aligned[i]
        
        # === RSI CONDITIONS (LOOSE for trades) ===
        rsi_oversold_strong = rsi_7[i] < 30.0
        rsi_oversold_loose = rsi_7[i] < 40.0
        rsi_overbought_strong = rsi_7[i] > 70.0
        rsi_overbought_loose = rsi_7[i] > 60.0
        
        # === SESSION FILTER (prefer 00-12 UTC but allow all for crypto) ===
        # London/NY overlap has higher volume, but crypto trades 24/7
        # We'll use session as a TIE-BREAKER, not hard filter
        in_preferred_session = (hours[i] >= 0 and hours[i] < 12)
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG conditions
        if htf_1d_bull and htf_1h_bull:
            if rsi_oversold_strong:
                desired_signal = SIZE_STRONG
            elif rsi_oversold_loose:
                # Reduce size if outside preferred session
                desired_signal = SIZE_BASE if in_preferred_session else SIZE_BASE * 0.8
        
        # SHORT conditions
        elif htf_1d_bear and htf_1h_bear:
            if rsi_overbought_strong:
                desired_signal = -SIZE_STRONG
            elif rsi_overbought_loose:
                # Reduce size if outside preferred session
                desired_signal = -SIZE_BASE if in_preferred_session else -SIZE_BASE * 0.8
        
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