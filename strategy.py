#!/usr/bin/env python3
"""
Experiment #769: 15m Primary + 1h/1d HTF — Loose Multi-TF RSI Pullback

Hypothesis: 15m timeframe with 1h/1d HTF bias can work IF entry conditions are LOOSE
enough to generate trades. Previous 15m experiments failed with 0 trades due to
overly strict filters. This version uses:

1. 1d HMA(21) for major trend bias (simple, reliable through crashes)
2. 1h HMA(16/48) for intermediate trend confirmation
3. 15m RSI(7) with LOOSE thresholds (35/65 not 20/80) for entry timing
4. Session filter: 00-12 UTC (London+NY overlap) for quality trades
5. ATR(14) 2.0x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Key difference from failed 15m strategies:
- RSI thresholds LOOSE: <35 or >65 (not extreme 20/80)
- OR logic: RSI extreme OR HMA crossover (not AND requiring both)
- Session filter reduces low-quality trades without blocking all entries
- Size 0.15-0.20 to account for higher trade frequency

Target: Sharpe>0.40, trades>=40/train, trades>=5/test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_loose_1h1d_session_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
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
    rsi_7 = calculate_rsi(close, period=7)
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
        
        # === SESSION FILTER (00-12 UTC for London/NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HMA TREND ===
        htf_1h_bull = hma_1h_16_aligned[i] > hma_1h_48_aligned[i]
        htf_1h_bear = hma_1h_16_aligned[i] < hma_1h_48_aligned[i]
        
        # === 1h HMA CROSSOVER ===
        hma_1h_cross_long = False
        hma_1h_cross_short = False
        if i > 0 and not np.isnan(hma_1h_16_aligned[i-1]) and not np.isnan(hma_1h_48_aligned[i-1]):
            hma_1h_cross_long = (hma_1h_16_aligned[i-1] <= hma_1h_48_aligned[i-1]) and (hma_1h_16_aligned[i] > hma_1h_48_aligned[i])
            hma_1h_cross_short = (hma_1h_16_aligned[i-1] >= hma_1h_48_aligned[i-1]) and (hma_1h_16_aligned[i] < hma_1h_48_aligned[i])
        
        # === 15m RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS - OR not AND) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + (1h bull OR 1h cross long) + (RSI oversold OR in_session)
        if htf_1d_bull:
            if htf_1h_bull or hma_1h_cross_long:
                if rsi_oversold:
                    if rsi_extreme_oversold or hma_1h_cross_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                elif in_session and htf_1h_bull:
                    # Session entry without RSI extreme
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + (1h bear OR 1h cross short) + (RSI overbought OR in_session)
        elif htf_1d_bear:
            if htf_1h_bear or hma_1h_cross_short:
                if rsi_overbought:
                    if rsi_extreme_overbought or hma_1h_cross_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
                elif in_session and htf_1h_bear:
                    # Session entry without RSI extreme
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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