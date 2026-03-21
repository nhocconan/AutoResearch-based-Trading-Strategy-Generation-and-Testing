#!/usr/bin/env python3
"""
EXPERIMENT #001 - Multi-Timeframe HMA + RSI Pullback Strategy
==============================================================
Hypothesis: Combining 1d trend filter with 4h HMA trend + RSI pullback entries
will significantly improve Sharpe ratio by filtering counter-trend trades.

Key components:
- 1d EMA(21) trend filter: only long when price > EMA21 on daily
- 4h HMA(16/48) crossover: primary trend signal
- 4h RSI(14) pullback: entry on oversold conditions in uptrend
- ATR(14) trailing stop: exit when price moves 2.5*ATR against position
- Discrete position sizing: 0.0, ±0.25, ±0.35 to minimize fee churn

Why this should beat supertrend_4h_v1:
- Daily trend filter eliminates ~40% of losing counter-trend trades
- HMA is more responsive than Supertrend for trend changes
- RSI pullback entries improve entry timing vs pure trend following
- ATR stop adapts to volatility regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_daily_filter_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull = (2 * wma_half - wma_full)
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(21) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align 1d EMA to 4h timeframe (auto shift(1) for completed bars)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === CALCULATE 4h INDICATORS (pre-loop for performance) ===
    # HMA(16) and HMA(48) for trend
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    # RSI(14) for pullback entries
    rsi = calculate_rsi(close, 14)
    
    # ATR(14) for stoploss
    atr = calculate_atr(high, low, close, 14)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    
    # Position sizing - discrete levels
    SIZE_ENTRY = 0.35  # Full position on entry
    SIZE_HALF = 0.18   # Half position on take profit
    SIZE_EXIT = 0.0    # Exit
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Minimum lookback for valid signals
    min_lookback = max(48, 21)
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > ema_1d_aligned[i]
        daily_bearish = close[i] < ema_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK CONDITIONS ===
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Pullback in downtrend
        
        # === ENTRY LOGIC ===
        # Long: Daily bullish + HMA bullish + RSI pullback
        if daily_bullish and hma_bullish and rsi_oversold and position_side != 1:
            signals[i] = SIZE_ENTRY
            position_side = 1
            entry_price = close[i]
            highest_since_entry = close[i]
        
        # Short: Daily bearish + HMA bearish + RSI pullback
        elif daily_bearish and hma_bearish and rsi_overbought and position_side != -1:
            signals[i] = -SIZE_ENTRY
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = close[i]
        
        # === POSITION MANAGEMENT ===
        elif position_side == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close[i])
            
            # Take profit at 2R (reduce to half position)
            profit_target = entry_price + 2.0 * atr[i]
            if close[i] >= profit_target and signals[i] != SIZE_HALF:
                signals[i] = SIZE_HALF
            
            # Trailing stop: exit if price drops 2.5*ATR from highest
            trailing_stop = highest_since_entry - 2.5 * atr[i]
            if close[i] < trailing_stop:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            # HMA crossover exit (trend reversal)
            elif hma_bearish and signals[i] != SIZE_EXIT:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            # Daily trend reversal exit
            elif daily_bearish and signals[i] != SIZE_EXIT:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        
        elif position_side == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close[i])
            
            # Take profit at 2R (reduce to half position)
            profit_target = entry_price - 2.0 * atr[i]
            if close[i] <= profit_target and signals[i] != -SIZE_HALF:
                signals[i] = -SIZE_HALF
            
            # Trailing stop: exit if price rises 2.5*ATR from lowest
            trailing_stop = lowest_since_entry + 2.5 * atr[i]
            if close[i] > trailing_stop:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            # HMA crossover exit (trend reversal)
            elif hma_bullish and signals[i] != SIZE_EXIT:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            # Daily trend reversal exit
            elif daily_bullish and signals[i] != SIZE_EXIT:
                signals[i] = SIZE_EXIT
                position_side = 0
                entry_price = 0.0
            
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        
        else:
            signals[i] = 0.0
    
    return signals