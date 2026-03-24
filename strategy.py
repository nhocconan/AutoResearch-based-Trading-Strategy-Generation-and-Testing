#!/usr/bin/env python3
"""
Experiment #929: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 1h/1d HTF bias can capture intraday moves while
avoiding whipsaws. Key is using HTF for DIRECTION (1d/1h HMA) and 15m only for
ENTRY TIMING (RSI pullback). Session filter (00-12 UTC) avoids low-volume periods.

Key innovations:
1. 1d HMA(21) for primary trend bias - price above = bullish, below = bearish
2. 1h HMA(16/48) for intermediate trend confirmation
3. 15m RSI(7) for entry timing - enter on pullback in trend direction
4. Session filter: prefer 00-12 UTC (London/NY overlap)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 to minimize fee churn (15m = higher freq)
7. LOOSE entry conditions to ensure ≥10 trades/train, ≥3/test (learned from #919/#921/#925)

Entry conditions (LOOSE to guarantee trades - learned from zero-trade failures):
- LONG = 1d HMA bull + (1h HMA bull OR RSI<35) + optional session
- SHORT = 1d HMA bear + (1h HMA bear OR RSI>65) + optional session
- RSI extremes (<20/>80) override session filter to ensure trade generation

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
Trade freq target: 50-100 trades/year (CRITICAL for 15m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_v1"
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
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
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1h_16_raw = calculate_hma(df_1h['close'].values, period=16)
    hma_1h_48_raw = calculate_hma(df_1h['close'].values, period=48)
    hma_1h_16_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_16_raw)
    hma_1h_48_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_48_raw)
    
    # Calculate 15m indicators
    hma_15m_16 = calculate_hma(close, period=16)
    hma_15m_48 = calculate_hma(close, period=48)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
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
        
        if np.isnan(hma_15m_16[i]) or np.isnan(hma_15m_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        # === INTERMEDIATE TREND (1h HMA) ===
        htf_1h_bull = hma_1h_16_aligned[i] > hma_1h_48_aligned[i]
        htf_1h_bear = hma_1h_16_aligned[i] < hma_1h_48_aligned[i]
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_15m_16[i] > hma_15m_48[i]
        hma_15m_bear = hma_15m_16[i] < hma_15m_48[i]
        
        # === RSI CONDITIONS (LOOSE TO GUARANTEE TRADES) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 20.0  # Override filter
        rsi_extreme_overbought = rsi_7[i] > 80.0  # Override filter
        
        # === SESSION FILTER (00-12 UTC) ===
        # Crypto high-volume period (London/NY overlap)
        hour_utc = (prices['open_time'].iloc[i] // 3600000) % 24
        in_session = 0 <= hour_utc < 12
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        # Learn from #919/#921/#925: zero trades = conditions too strict
        desired_signal = 0.0
        
        # LONG entries (HTF bullish bias)
        if htf_1d_bull:
            # Strong: all 3 trends align + RSI pullback
            if htf_1h_bull and hma_15m_bull and rsi_oversold:
                if in_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Medium: 1d + 1h bull + RSI pullback (relaxed 15m trend)
            elif htf_1h_bull and rsi_oversold:
                if in_session:
                    desired_signal = SIZE_BASE
            # Weak: 1d bull + extreme RSI (override session)
            elif rsi_extreme_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT entries (HTF bearish bias)
        elif htf_1d_bear:
            # Strong: all 3 trends align + RSI pullback
            if htf_1h_bear and hma_15m_bear and rsi_overbought:
                if in_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Medium: 1d + 1h bear + RSI pullback (relaxed 15m trend)
            elif htf_1h_bear and rsi_overbought:
                if in_session:
                    desired_signal = -SIZE_BASE
            # Weak: 1d bear + extreme RSI (override session)
            elif rsi_extreme_overbought:
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