#!/usr/bin/env python3
"""
Experiment #337: 15m Primary + 4h/1d HTF — Session-Aware HMA/RSI with Regime Filter

Hypothesis: 15m strategies failed with 0 trades due to overly strict entry conditions.
This strategy uses LOOSER thresholds while maintaining quality via HTF confluence.

Key design:
1. 4h HMA(21) for trend direction (aligned properly via mtf_data)
2. 1d HMA(50) for major bias filter
3. 15m RSI(7) for entry timing — pullback entries in trend direction
4. 4h Choppiness Index to switch between trend-follow vs mean-revert mode
5. Session filter: prefer 00-12 UTC (London/NY overlap) but allow strong signals outside
6. Position sizing: 0.20 base, 0.30 when 4h+1d aligned
7. Stoploss: 2.5x ATR(14) from entry

Why this should work on 15m:
- RSI(7) is faster than RSI(14), catches more pullbacks
- 4h regime filter prevents trading against higher timeframe
- Session filter reduces noise but doesn't block all trades
- LOOSENED thresholds: RSI < 40 / > 60 (not 30/70) to ensure trades generate

Target: 50-100 trades/year, Sharpe > 0.40, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_hma_rsi_chop_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

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
    
    # Calculate 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    hma_15m_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_std = calculate_rsi(close, period=14)
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Regime memory
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_std[i]):
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
        
        if np.isnan(chop_4h_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # Extract hour from open_time (milliseconds since epoch)
        hour_utc = (open_time[i] // 3600000) % 24
        is_preferred_session = 0 <= hour_utc <= 12
        
        # === 4H REGIME DETECTION ===
        choppy_threshold = 58.0
        trending_threshold = 45.0
        
        if chop_4h_aligned[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop_4h_aligned[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_15m_fast[i]) and not np.isnan(hma_15m_fast[i-1]):
            if not np.isnan(hma_15m[i]) and not np.isnan(hma_15m[i-1]):
                if hma_15m_fast[i-1] <= hma_15m[i-1] and hma_15m_fast[i] > hma_15m[i]:
                    hma_cross_long = True
                if hma_15m_fast[i-1] >= hma_15m[i-1] and hma_15m_fast[i] < hma_15m[i]:
                    hma_cross_short = True
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        rsi_oversold = rsi_fast[i] < 42.0  # Was 30, now 42 for more signals
        rsi_overbought = rsi_fast[i] > 58.0  # Was 70, now 58 for more signals
        
        rsi_extreme_oversold = rsi_std[i] < 35.0
        rsi_extreme_overbought = rsi_std[i] > 65.0
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.001
        bb_touch_upper = close[i] >= bb_upper[i] * 0.999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (follow 4h trend with 15m pullback entries)
        if current_regime == 1:
            # Long: 4h bull + 15m HMA bull + RSI pullback oversold
            if htf_4h_bull and hma_bull and rsi_oversold:
                size_mult = 1.5 if (htf_1d_bull and is_preferred_session) else 1.0
                desired_signal = SIZE_BASE * size_mult
            
            # Short: 4h bear + 15m HMA bear + RSI pullback overbought
            elif htf_4h_bear and hma_bear and rsi_overbought:
                size_mult = 1.5 if (htf_1d_bear and is_preferred_session) else 1.0
                desired_signal = -SIZE_BASE * size_mult
            
            # HMA crossover entry (stronger signal)
            elif hma_cross_long and htf_4h_bull:
                size_mult = 1.5 if htf_1d_bull else 1.0
                desired_signal = SIZE_BASE * size_mult
            
            elif hma_cross_short and htf_4h_bear:
                size_mult = 1.5 if htf_1d_bear else 1.0
                desired_signal = -SIZE_BASE * size_mult
        
        # REGIME 2: CHOPPY (mean reversion at Bollinger extremes)
        elif current_regime == 2:
            # Long: BB lower touch + RSI oversold + 4h not strongly bearish
            if bb_touch_lower and rsi_extreme_oversold and not htf_4h_bear:
                desired_signal = SIZE_BASE
            
            # Short: BB upper touch + RSI overbought + 4h not strongly bullish
            elif bb_touch_upper and rsi_extreme_overbought and not htf_4h_bull:
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