#!/usr/bin/env python3
"""
EXPERIMENT #019 - MACD + Bollinger Breakout with 4h Trend Filter (15m)
=======================================================================
Hypothesis: 15m MACD histogram momentum combined with Bollinger Band breakouts
captures short-term trend continuations when aligned with 4h HMA(21) trend.
Bollinger Band squeeze (low bandwidth) precedes explosive moves. Volume
confirms breakout validity. ATR trailing stop protects against reversals.

Why this differs from failed strategies:
- Most failed strategies used HMA/RSI pullback (overused pattern)
- This uses MACD momentum + BB volatility breakout (different signal type)
- 15m TF provides more trade opportunities than 1h/4h/daily
- 4h HMA filter prevents counter-trend trades (reduces drawdown)

Key features:
- Primary TF: 15m (faster entries, more trades)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: MACD histogram turning + BB breakout + volume spike
- Filter: 4h trend must align with entry direction
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.35 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_bb_4h_trend_15m_v1"
timeframe = "15m"
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


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    return upper.values, lower.values, sma.values, bandwidth.values


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
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Band squeeze detection (bandwidth below 20-period median)
    bb_bandwidth_sma = pd.Series(bb_bandwidth).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.30  # Base position size (30% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for 4h HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or 
            np.isnan(rsi[i]) or atr[i] == 0 or np.isnan(bb_bandwidth_sma[i])):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > 1.2 * volume_sma[i]  # 20% above average
        
        # Bollinger Band squeeze (low volatility before breakout)
        bb_squeeze = bb_bandwidth[i] < 0.8 * bb_bandwidth_sma[i]
        
        # MACD momentum signal
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i - 1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i - 1]
        
        # MACD histogram turning point (momentum shift)
        macd_turn_long = macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] <= 0
        macd_turn_short = macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] >= 0
        
        # Bollinger breakout detection
        bb_breakout_long = close[i] > bb_upper[i] and close[i - 1] <= bb_upper[i - 1]
        bb_breakout_short = close[i] < bb_lower[i] and close[i - 1] >= bb_lower[i - 1]
        
        # RSI filter (avoid extreme overbought/oversold)
        rsi_valid_long = rsi[i] < 70
        rsi_valid_short = rsi[i] > 30
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h uptrend + MACD bullish turn + BB breakout + volume
        if (hma_trend == 1 and macd_turn_long and bb_breakout_long and 
            volume_confirmed and rsi_valid_long):
            target_signal = SIZE
        
        # Short entry: 4h downtrend + MACD bearish turn + BB breakout + volume
        elif (hma_trend == -1 and macd_turn_short and bb_breakout_short and 
              volume_confirmed and rsi_valid_short):
            target_signal = -SIZE
        
        # Alternative: MACD momentum continuation (no breakout needed if strong trend)
        elif (hma_trend == 1 and macd_bullish and close[i] > bb_sma[i] and 
              volume_confirmed and rsi_valid_long and position_side == 0):
            target_signal = SIZE * 0.8  # Slightly smaller for momentum-only entries
        
        elif (hma_trend == -1 and macd_bearish and close[i] < bb_sma[i] and 
              volume_confirmed and rsi_valid_short and position_side == 0):
            target_signal = -SIZE * 0.8
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    risk = entry_price - (entry_price - 2.0 * entry_atr)  # 2*ATR risk
                    if close[i] >= entry_price + 2.0 * risk:  # 2R profit = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    risk = (entry_price + 2.0 * entry_atr) - entry_price  # 2*ATR risk
                    if close[i] <= entry_price - 2.0 * risk:  # 2R profit
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
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    entry_atr = atr[i]
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals