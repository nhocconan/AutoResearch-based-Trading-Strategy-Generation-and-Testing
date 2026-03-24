#!/usr/bin/env python3
"""
Experiment #077: 15m Primary + 4h/1d HTF — RSI Mean Reversion + HMA Trend + Session Filter

Hypothesis: After 76 failed experiments, 15m strategies keep returning 0 trades because
entry conditions are TOO STRICT. This strategy uses LOOSE entry filters to ensure
trades generate while maintaining HTF bias for direction.

Key design choices:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 4h HMA(21) for trend bias, 1d HMA(50) for major regime
- Entry: RSI(7) extremes (<30 long, >70 short) — very responsive on 15m
- Vol filter: ATR(7)/ATR(30) > 0.7 to avoid dead markets
- Session: Prefer 00-12 UTC (London+NY overlap) but allow 12-24 with stronger RSI
- Position size: 0.20 (conservative for 15m frequency)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: Only require HTF bias + RSI extreme (not all conditions)

Why this should work:
- RSI(7) on 15m generates frequent signals (unlike RSI(14) which is too slow)
- HTF bias prevents counter-trend trades but doesn't block entries
- Session filter reduces trades during low-volume periods (Asia session)
- ATR vol filter avoids choppy dead markets
- Position size 0.20 limits drawdown on 15m whipsaws

Target: Sharpe>0.167, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_session_4h1d_v1"
timeframe = "15m"
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
    """Average True Range for stoploss and vol filter"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY FILTER (avoid dead markets) ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_ok = atr_ratio > 0.7  # Market has some volatility
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF MAJOR REGIME (1d HMA) ===
        major_bull = close[i] > hma_1d_aligned[i]
        major_bear = close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_prime_session = 0 <= hour_utc < 12  # London+NY overlap preferred
        
        # === RSI EXTREMES (15m) ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        rsi_extreme_oversold = rsi_7[i] < 20.0
        rsi_extreme_overbought = rsi_7[i] > 80.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entries (mean reversion in uptrend)
        if vol_ok:
            # Prime session: looser RSI requirement
            if is_prime_session:
                if rsi_oversold and htf_bull:
                    desired_signal = SIZE
                elif rsi_extreme_oversold:  # Strong oversold, ignore HTF
                    desired_signal = SIZE * 0.7
            # Off-peak session: require stronger signals
            else:
                if rsi_extreme_oversold and htf_bull:
                    desired_signal = SIZE
                elif rsi_extreme_oversold and major_bull:  # Major trend support
                    desired_signal = SIZE * 0.7
        
        # SHORT entries (mean reversion in downtrend)
        if vol_ok:
            # Prime session: looser RSI requirement
            if is_prime_session:
                if rsi_overbought and htf_bear:
                    desired_signal = -SIZE
                elif rsi_extreme_overbought:  # Strong overbought, ignore HTF
                    desired_signal = -SIZE * 0.7
            # Off-peak session: require stronger signals
            else:
                if rsi_extreme_overbought and htf_bear:
                    desired_signal = -SIZE
                elif rsi_extreme_overbought and major_bear:  # Major trend support
                    desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals