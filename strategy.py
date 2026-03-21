#!/usr/bin/env python3
"""
EXPERIMENT #021 - MACD Momentum with 4h HMA Trend + Volume Filter (1h)
========================================================================
Hypothesis: 1h MACD histogram momentum entries aligned with 4h HMA(50) trend
direction capture sustained moves while avoiding counter-trend traps. Volume
confirmation filters false breakouts. ATR trailing stop protects capital.
This differs from previous RSI pullback approaches by using momentum confirmation
rather than mean reversion logic.

Key features:
- Primary TF: 1h (hourly candles)
- HTF filter: 4h HMA(50) for trend direction
- Entry: MACD(12,26,9) histogram turning positive/negative + volume confirmation
- Filter: 4h trend must align with entry direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 discrete levels (conservative)
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_momentum_4h_trend_1h_v1"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_200 = calculate_ema(close, 200)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 250  # Wait for 4h HMA, EMA200, and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or np.isnan(rsi[i]) or
            np.isnan(ema_200[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF direction)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h EMA200 filter (secondary trend confirmation)
        ema_trend = 1 if close[i] > ema_200[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # MACD histogram momentum signal
        macd_signal_dir = 0
        if i > 0:
            # Long: histogram turning positive (crossing above zero or increasing)
            if macd_hist[i] > 0 and macd_hist[i - 1] <= 0:
                macd_signal_dir = 1
            # Short: histogram turning negative (crossing below zero or decreasing)
            elif macd_hist[i] < 0 and macd_hist[i - 1] >= 0:
                macd_signal_dir = -1
            # Alternative: histogram momentum (increasing/decreasing)
            elif macd_hist[i] > macd_hist[i - 1] and macd_hist[i] > 0:
                macd_signal_dir = 1
            elif macd_hist[i] < macd_hist[i - 1] and macd_hist[i] < 0:
                macd_signal_dir = -1
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_valid_long = rsi[i] < 70  # Not overbought for long entry
        rsi_valid_short = rsi[i] > 30  # Not oversold for short entry
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        if macd_signal_dir == 1:
            # Long entry: MACD bullish + 4h trend up + EMA200 up + volume + RSI valid
            if hma_trend == 1 and ema_trend == 1 and volume_confirmed and rsi_valid_long:
                target_signal = SIZE
        elif macd_signal_dir == -1:
            # Short entry: MACD bearish + 4h trend down + EMA200 down + volume + RSI valid
            if hma_trend == -1 and ema_trend == -1 and volume_confirmed and rsi_valid_short:
                target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    risk_distance = 2.5 * entry_atr
                    profit_target = entry_price + 2.0 * risk_distance
                    if close[i] >= profit_target:
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    risk_distance = 2.5 * entry_atr
                    profit_target = entry_price - 2.0 * risk_distance
                    if close[i] <= profit_target:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Check if we should reverse position
                if target_signal != 0.0 and np.sign(target_signal) != position_side:
                    # Reverse: close current and open new
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    entry_atr = atr[i]
                    profit_target_hit = False
                else:
                    # Maintain existing position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals