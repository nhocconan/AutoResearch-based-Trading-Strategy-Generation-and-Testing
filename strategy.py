#!/usr/bin/env python3
"""
Experiment #457: 15m Primary + 4h HTF — Simplified Trend Pullback

Hypothesis: 15m strategies failing with 0 trades due to OVER-FILTERING.
Recent 15m failures (#445, #449, #453, #456) all have Sharpe=0.000 = 0 trades.

Key changes from failed attempts:
1. SINGLE HTF (4h only, not 4h+12h dual) - dual is too restrictive for 15m
2. FASTER RSI (period=7 instead of 14) - 15m needs quicker signals
3. NO SESSION FILTER - session filters killed all trades in prior 15m attempts
4. SIMPLER ENTRY: RSI extreme + HTF trend alignment (just 2 conditions)
5. LOOSER RSI THRESHOLDS: 25/75 instead of 20/80 (more trades qualify)
6. SIZE: 0.20 base (smaller for 15m frequency, target 50-100 trades/year)

Entry Logic:
- Long: 4h HMA bull + 15m RSI(7) < 25 (oversold pullback in uptrend)
- Short: 4h HMA bear + 15m RSI(7) > 75 (overbought pullback in downtrend)
- Exit: RSI crosses 50 OR stoploss hit (2.5x ATR)

Why this should work:
- 4h trend filter prevents counter-trend trades (major edge)
- RSI(7) extremes on 15m happen frequently enough for trades
- No session filter = trades throughout day
- Simple logic = fewer conditions that can all fail simultaneously

Target: Sharpe>0.4, DD>-35%, trades>=100 train (25/year), trades>=15 test
Timeframe: 15m (FIRST working 15m strategy after 4 failures)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_trend_rsi7_pullback_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    atr = calculate_atr(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)  # Dynamic S/R
    ema_50 = calculate_ema(close, period=50)  # Trend confirmation
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Cooldown to prevent rapid re-entry
    last_exit_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m TREND CONFIRMATION ===
        trend_bull = ema_21[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        trend_bear = ema_21[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # === RSI EXTREMES (FASTER: period=7, thresholds=25/75) ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        
        # RSI confirmation (standard period)
        rsi_14_oversold = rsi_14[i] < 35.0
        rsi_14_overbought = rsi_14[i] > 65.0
        
        # === PRICE POSITION vs EMA ===
        above_ema21 = close[i] > ema_21[i]
        below_ema21 = close[i] < ema_21[i]
        
        # === ENTRY LOGIC (SIMPLE: 2-3 conditions max) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI(7) oversold + price near/above EMA21
        if htf_bull and rsi_oversold:
            # At least one confirmation needed
            if above_ema21 or rsi_14_oversold or trend_bull:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m RSI(7) overbought + price near/below EMA21
        elif htf_bear and rsi_overbought:
            # At least one confirmation needed
            if below_ema21 or rsi_14_overbought or trend_bear:
                desired_signal = -SIZE_BASE
        
        # === STRONGER ENTRY (all conditions align) ===
        if htf_bull and rsi_oversold and trend_bull and above_ema21:
            desired_signal = SIZE_STRONG
        elif htf_bear and rsi_overbought and trend_bear and below_ema21:
            desired_signal = -SIZE_STRONG
        
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
            last_exit_bar = i
        
        # === RSI EXIT (cross back through 50) ===
        if in_position and i > 0:
            if position_side > 0 and rsi_7[i] > 55.0:  # Long exit on RSI recovery
                desired_signal = 0.0
                last_exit_bar = i
            elif position_side < 0 and rsi_7[i] < 45.0:  # Short exit on RSI decline
                desired_signal = 0.0
                last_exit_bar = i
        
        # === COOLDOWN CHECK (prevent rapid re-entry after exit) ===
        if i - last_exit_bar < 5:  # 5 bar cooldown (75 minutes on 15m)
            if desired_signal != 0.0 and in_position == False:
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