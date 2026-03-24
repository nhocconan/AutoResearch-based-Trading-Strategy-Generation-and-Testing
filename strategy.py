#!/usr/bin/env python3
"""
Experiment #540: 6h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 6h strategies failed due to overly complex filters (ADX+CHOP+CRSI)
that blocked all trades. This strategy uses SIMPLER logic:
1. 1w HMA(21) = macro bias (only direction filter, not entry trigger)
2. 1d HMA(21) = medium bias (confirms trend direction)
3. 6h HMA(16/48) crossover = primary entry signal
4. 6h RSI(14) = timing filter (oversold in uptrend, overbought in downtrend)
5. ATR(14)*2.5 stoploss on all positions

Key differences from failed #535 (mtf_6h_crsi_hma_chop_regime):
1. REMOVED ADX filter (was blocking trades)
2. REMOVED CHOP filter (was blocking trades)
3. REMOVED CRSI (overcomplicated, use simple RSI)
4. LOOSENED RSI thresholds (30/70 instead of 25/75)
5. Added HMA crossover entry (more reliable than price vs HMA)

Why this should work:
- 6h is middle ground between 4h (too many trades) and 12h (too few)
- HMA crossover captures trend changes with less lag than EMA
- HTF bias prevents counter-trend trades
- Simple RSI filter ensures entries at pullbacks, not chasing
- Target: 40-80 trades/year (20-40 per year per symbol)

Strategy logic:
1. Load 1d and 1w data ONCE before loop (Rule 1)
2. Calculate HMA on all timeframes
3. Align HTF HMA to 6h using align_htf_to_ltf (Rule 2)
4. Entry: HMA16 crosses above HMA48 + RSI>40 + 1d/1w bullish
5. Exit: HMA16 crosses below HMA48 OR stoploss hit
6. Position size: 0.25-0.30 (discrete levels)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_1d1w_v2"
timeframe = "6h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        # Bullish: price above both HTF HMAs
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        # Bearish: price below both HTF HMAs
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        # Neutral: mixed signals
        htf_neutral = not htf_bull and not htf_bear
        
        # === HMA CROSSOVER (primary entry signal) ===
        # Fast HMA above slow HMA = bullish trend
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # HMA crossover detection (fast crosses above/below slow)
        hma_cross_bull = False
        hma_cross_bear = False
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            hma_cross_bull = (hma_fast[i-1] <= hma_slow[i-1]) and (hma_fast[i] > hma_slow[i])
            hma_cross_bear = (hma_fast[i-1] >= hma_slow[i-1]) and (hma_fast[i] < hma_slow[i])
        
        # === RSI FILTER (pullback entries) ===
        # In uptrend: enter on RSI pullback (not extreme)
        rsi_pullback_long = rsi[i] > 35.0 and rsi[i] < 60.0
        # In downtrend: enter on RSI bounce (not extreme)
        rsi_pullback_short = rsi[i] > 40.0 and rsi[i] < 65.0
        # RSI extreme for stronger conviction
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === HMA SLOPE (trend confirmation) ===
        hma_slope_bull = False
        hma_slope_bear = False
        if i >= 5 and not np.isnan(hma_fast[i-5]):
            hma_slope_bull = hma_fast[i] > hma_fast[i-5]
            hma_slope_bear = hma_fast[i] < hma_fast[i-5]
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull or (not htf_bear):  # Allow long if not clearly bearish
            # HMA crossover + RSI confirmation
            if hma_cross_bull and rsi_pullback_long:
                desired_signal = SIZE_STRONG
            # HMA already bullish + RSI pullback + slope confirmation
            elif hma_bull and hma_slope_bull and rsi_oversold:
                desired_signal = SIZE_BASE
            # HMA bullish + RSI recovering from oversold
            elif hma_bull and rsi[i] > 30.0 and i > 0 and rsi[i] > rsi[i-1]:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        if htf_bear or (not htf_bull):  # Allow short if not clearly bullish
            # HMA crossover + RSI confirmation
            if hma_cross_bear and rsi_pullback_short:
                desired_signal = -SIZE_STRONG
            # HMA already bearish + RSI bounce + slope confirmation
            elif hma_bear and hma_slope_bear and rsi_overbought:
                desired_signal = -SIZE_BASE
            # HMA bearish + RSI rolling over from overbought
            elif hma_bear and rsi[i] < 70.0 and i > 0 and rsi[i] < rsi[i-1]:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON HMA REVERSAL ===
        if in_position and position_side > 0 and hma_cross_bear:
            desired_signal = 0.0
        if in_position and position_side < 0 and hma_cross_bull:
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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