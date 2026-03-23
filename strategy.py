#!/usr/bin/env python3
"""
Experiment #151: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Recent failures show overly complex regime detection = poor results.
Going back to proven pattern: HMA trend + RSI pullback entries (worked in best strategy).

Key changes from #141:
1) SIMPLER logic: HMA(21/63) crossover for trend, RSI(14) for pullback entries
2) FIX position tracking: hold positions properly without premature exits
3) MORE lenient entries: RSI 35-45 long, 55-65 short (not extreme 20/80)
4) 1d HMA(21) for macro bias only (not hard filter)
5) ATR(14) trailing stop at 2.5x — protects capital
6) Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols

Why this should work:
- HMA crossover proven in trend following (less lag than EMA)
- RSI pullback entries catch dips in uptrend (not chasing breakouts)
- Simpler = more reliable signals, fewer conflicting filters
- 4h naturally produces 25-50 trades/year (low fee drag)
- Fixed position tracking = proper trade duration

Position size: 0.25 base, 0.30 with confluence
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50).values

def calculate_hma_crossover(hma_fast, hma_slow):
    """Detect HMA crossover direction."""
    crossover = np.zeros(len(hma_fast))
    for i in range(1, len(hma_fast)):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        # Bullish crossover: fast crosses above slow
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            crossover[i] = 1.0
        # Bearish crossover: fast crosses below slow
        elif hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            crossover[i] = -1.0
    return crossover

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    rsi_14 = calculate_rsi(close, period=14)
    
    # HMA crossover signal
    hma_cross = calculate_hma_crossover(hma_21, hma_63)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 999999.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: HMA bullish + RSI pullback + 1d bias supportive
        if hma_bullish and rsi_oversold:
            if price_above_hma_1d or hma_cross[i] == 1.0:
                new_signal = POSITION_SIZE_BASE
                if rsi_14[i] < 35.0:
                    new_signal = POSITION_SIZE_MAX
        
        # SHORT ENTRY: HMA bearish + RSI pullback + 1d bias supportive
        if hma_bearish and rsi_overbought:
            if price_below_hma_1d or hma_cross[i] == -1.0:
                new_signal = -POSITION_SIZE_BASE
                if rsi_14[i] > 65.0:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # If we have a position, hold it unless exit conditions met
        if position_side != 0 and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL (HMA cross against position) ===
        if position_side > 0 and hma_bearish and rsi_14[i] > 50.0:
            new_signal = 0.0
        
        if position_side < 0 and hma_bullish and rsi_14[i] < 50.0:
            new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0 or np.sign(new_signal) != position_side:
                # New position or flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 999999.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if position_side != 0 and signals[i-1] != 0.0:
                # Position closed - reset tracking
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 999999.0
        
        signals[i] = new_signal
    
    return signals