#!/usr/bin/env python3
"""
Experiment #737: 15m Primary + 4h/12h HTF — Loose Pullback Strategy

Hypothesis: 15m timeframe with dual HTF (4h + 12h) trend filters can capture
intraday pullbacks within larger trends. Previous 15m experiments failed with
0 trades due to overly strict conditions. This version uses LOOSE RSI thresholds
(35/65 not 20/80) and allows entry on either HTF confirmation.

Key innovations:
1. 12h HMA(21) for major trend bias — call ONCE before loop
2. 4h HMA(16/48) for intermediate trend — call ONCE before loop
3. 15m RSI(7) with loose thresholds (35/65) for pullback entries
4. Session filter: prefer 00-12 UTC (London/NY overlap)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 12h HMA bull + 4h HMA bull + RSI(7) < 40 OR RSI(7) crosses above 35
- SHORT: 12h HMA bear + 4h HMA bear + RSI(7) > 60 OR RSI(7) crosses below 65

Target: Sharpe>0.40, trades>=10 train, trades>=3 test, DD>-40%, trades<100/year
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h12h_loose_v1"
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA (12h major trend)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align HTF HMA crossover (4h intermediate trend)
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
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
    
    for i in range(50, n):
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA - major trend) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h HMA CROSSOVER (intermediate trend) ===
        htf_4h_bull = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        htf_4h_bear = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 4h HMA CROSSOVER SIGNAL (for entry trigger) ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_4h_16_aligned[i-1]) and not np.isnan(hma_4h_48_aligned[i-1]):
            hma_crossover_long = (hma_4h_16_aligned[i-1] <= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] > hma_4h_48_aligned[i])
            hma_crossover_short = (hma_4h_16_aligned[i-1] >= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] < hma_4h_48_aligned[i])
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_7[i] < 40.0
        rsi_overbought = rsi_7[i] > 60.0
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # RSI crossing up from oversold
        rsi_cross_up = False
        rsi_cross_down = False
        if i > 0 and not np.isnan(rsi_7[i-1]):
            rsi_cross_up = (rsi_7[i-1] < 35.0) and (rsi_7[i] >= 35.0)
            rsi_cross_down = (rsi_7[i-1] > 65.0) and (rsi_7[i] <= 65.0)
        
        # === SESSION FILTER (prefer 00-12 UTC for liquidity) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        prime_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull (12h OR 4h) + RSI pullback
        if htf_12h_bull or htf_4h_bull:
            # Strong signal: extreme oversold or crossover
            if rsi_extreme_oversold or hma_crossover_long:
                if prime_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Base signal: regular oversold or RSI cross up
            elif rsi_oversold or rsi_cross_up:
                if prime_session:
                    desired_signal = SIZE_BASE
                else:
                    desired_signal = SIZE_BASE * 0.5
        
        # SHORT: HTF bear (12h OR 4h) + RSI pullback
        elif htf_12h_bear or htf_4h_bear:
            # Strong signal: extreme overbought or crossover
            if rsi_extreme_overbought or hma_crossover_short:
                if prime_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Base signal: regular overbought or RSI cross down
            elif rsi_overbought or rsi_cross_down:
                if prime_session:
                    desired_signal = -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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