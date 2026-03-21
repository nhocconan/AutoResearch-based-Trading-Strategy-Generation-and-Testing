#!/usr/bin/env python3
"""
EXPERIMENT #003 - HMA Trend + MACD Momentum + RSI Pullback (1h)
================================================================
Hypothesis: 1h timeframe provides optimal balance between signal frequency and noise.
Combining 4h HMA(21) trend filter with 1h MACD momentum and RSI pullback entries
captures trend continuations at favorable entry points. This differs from failed
strategies by using MACD histogram for momentum confirmation instead of just RSI,
and HMA instead of Supertrend/KAMA for smoother trend detection.

Key features:
- Primary TF: 1h (hourly candles)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: MACD histogram turning positive/negative + RSI pullback (40-60 zone)
- Filter: 4h trend must align with entry direction
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_macd_rsi_pullback_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h[i]) or 
            np.isnan(atr[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF)
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h trend confirmation
        hma_1h_trend = 1 if close[i] > hma_1h[i] else -1
        
        # MACD momentum signal
        macd_signal_dir = 0
        if i > 0:
            if macd_hist[i] > 0 and macd_hist[i - 1] <= 0:
                macd_signal_dir = 1  # Bullish crossover
            elif macd_hist[i] < 0 and macd_hist[i - 1] >= 0:
                macd_signal_dir = -1  # Bearish crossover
        
        # RSI pullback filter (entry in neutral zone, not extreme)
        rsi_valid_long = 40 <= rsi[i] <= 65
        rsi_valid_short = 35 <= rsi[i] <= 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend up + 1h trend up + MACD bullish + RSI valid
        if macd_signal_dir == 1 and hma_4h_trend == 1 and hma_1h_trend == 1 and rsi_valid_long:
            target_signal = SIZE
        
        # Short entry: 4h trend down + 1h trend down + MACD bearish + RSI valid
        elif macd_signal_dir == -1 and hma_4h_trend == -1 and hma_1h_trend == -1 and rsi_valid_short:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Only enter if flat or reversing (avoid churning)
                if position_side == 0 or np.sign(target_signal) != position_side:
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
                else:
                    # Maintain existing position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals