#!/usr/bin/env python3
"""
EXPERIMENT #001 - Supertrend + RSI Pullback with 4h Trend Filter (15m)
=======================================================================
Hypothesis: 15m Supertrend captures short-term momentum, RSI pullback entries
avoid chasing tops/bottoms, and 4h HMA(21) filter ensures we trade with the
higher timeframe trend. This multi-timeframe approach should improve Sharpe
vs single-TF strategies.

Key features:
- Primary TF: 15m (faster entries than 1h/4h strategies)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: Supertrend flip + RSI(14) pullback to 40-60 zone
- Filter: Only long if price > 4h HMA, only short if price < 4h HMA
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (max 0.35)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4hfilter_15m_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, n):
        if trend[i-1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(supertrend[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Supertrend signal
        st_signal = st_trend[i]
        st_prev = st_trend[i-1] if i > 0 else st_signal
        
        # Detect Supertrend flip
        st_flip = 0
        if st_signal == 1 and st_prev == -1:
            st_flip = 1  # Bullish flip
        elif st_signal == -1 and st_prev == 1:
            st_flip = -1  # Bearish flip
        
        # RSI pullback filter (avoid entering at extremes)
        # For longs: RSI should be 40-60 (pullback, not overbought)
        # For shorts: RSI should be 40-60 (pullback, not oversold)
        rsi_valid_long = 40 < rsi[i] < 65
        rsi_valid_short = 35 < rsi[i] < 60
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: Supertrend flip bullish + 4h trend up + RSI valid
        if st_flip == 1 and hma_trend == 1 and rsi_valid_long:
            target_signal = BASE_SIZE
        
        # Short entry: Supertrend flip bearish + 4h trend down + RSI valid
        elif st_flip == -1 and hma_trend == -1 and rsi_valid_short:
            target_signal = -BASE_SIZE
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Check if this is a reversal (costly) or same direction
                if position_side == 0 or np.sign(target_signal) == position_side:
                    signals[i] = target_signal
                    if position_side == 0:
                        # New entry
                        position_side = 1 if target_signal > 0 else -1
                        highest_since_entry = close[i]
                        lowest_since_entry = close[i]
                        entry_price = close[i]
                else:
                    # Reversal - flatten first, then enter (two signal changes)
                    # For simplicity, just switch (costs fees but cleaner code)
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = BASE_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals