#!/usr/bin/env python3
"""
Experiment #779: 1h Primary + 4h/12h HTF — Triple Confluence with Loose Thresholds

Hypothesis: 1h timeframe with 4h trend bias + RSI momentum + volatility filter
can achieve 40-80 trades/year with positive Sharpe. Previous 1h experiments
failed due to either 0 trades (too strict) or negative Sharpe (too many trades).

Key innovations:
1. 4h HMA(21) for HTF trend bias — proven reliable in best strategies
2. 1h RSI(10) with LOOSE thresholds (35/65 not 20/80) to ensure trade generation
3. 1h ATR ratio filter (ATR/ATR_SMA50 > 0.8) to avoid dead volatility periods
4. 12h HMA(16/48) crossover as additional confirmation (optional boost)
5. Session filter: 08-20 UTC only (reduces noise, focuses on liquid hours)
6. Discrete sizing: 0.0, ±0.20, ±0.25 — minimizes fee churn
7. 2.5x ATR trailing stoploss for risk management

Entry conditions (LOOSE to ensure ≥40 trades/train, ≥3/test):
- LONG: 4h HMA bull + RSI<40 + ATR_ratio>0.8 + (optional: 12h HMA bull)
- SHORT: 4h HMA bear + RSI>60 + ATR_ratio>0.8 + (optional: 12h HMA bear)

Target: Sharpe>0.40, trades>=40 train, trades>=3 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma_loose_4h12h_session_v1"
timeframe = "1h"
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
    """Average True Range - volatility measure"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_16_raw = calculate_hma(df_12h['close'].values, period=16)
    hma_12h_48_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_16_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_16_raw)
    hma_12h_48_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_48_raw)
    
    # Calculate 1h indicators
    rsi_10 = calculate_rsi(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio: current ATR / SMA of ATR(50)
    atr_sma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_sma50 + 1e-10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_16_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 8) and (hour_utc <= 20)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 12h HMA CROSSOVER ===
        hma_12h_bull = hma_12h_16_aligned[i] > hma_12h_48_aligned[i]
        hma_12h_bear = hma_12h_16_aligned[i] < hma_12h_48_aligned[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_10[i] < 40.0
        rsi_overbought = rsi_10[i] > 60.0
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] > 0.8
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + RSI oversold + vol ok + session
        if htf_4h_bull and rsi_oversold and vol_ok:
            if in_session:
                # Strong signal if 12h also bull
                if hma_12h_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + RSI overbought + vol ok + session
        elif htf_4h_bear and rsi_overbought and vol_ok:
            if in_session:
                # Strong signal if 12h also bear
                if hma_12h_bear:
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