#!/usr/bin/env python3
"""
EXPERIMENT #001 - Multi-Timeframe HMA + RSI Pullback Strategy
==============================================================
Hypothesis: Combining 1d trend filter with 4h HMA trend + RSI pullback entries 
will reduce whipsaws and improve risk-adjusted returns. The daily trend filter 
prevents counter-trend trades during major reversals, while RSI pullbacks provide 
better entry timing within the trend. ATR trailing stoplimits drawdown.

Key features:
- 1d EMA(21/55) trend filter (HTF) - loaded ONCE before loop
- 4h HMA(16/48) trend direction (primary)
- RSI(14) pullback entry (30-70 range for entries)
- ATR(14) trailing stoploss (2*ATR)
- Discrete position sizing (0.0, ±0.25, ±0.35)
- leverage = 1.0
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_pullback_v2"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    atr = np.nan_to_num(atr, nan=atr[period])
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA trend
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55_1d = pd.Series(df_1d['close'].values).ewm(span=55, min_periods=55, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe (auto shift(1) for completed bars)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Track position for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start after warmup
        # Skip if HTF data not available (NaN from alignment)
        if np.isnan(ema_21_aligned[i]) or np.isnan(ema_55_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend filter
        daily_bullish = ema_21_aligned[i] > ema_55_aligned[i]
        daily_bearish = ema_21_aligned[i] < ema_55_aligned[i]
        
        # 4h HMA trend
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # RSI pullback conditions
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Entry logic - only enter when no position
        if position_side == 0:
            # Long entry: daily bullish + 4h bullish + RSI pullback
            if daily_bullish and hma_bullish and rsi_oversold:
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
            
            # Short entry: daily bearish + 4h bearish + RSI pullback
            elif daily_bearish and hma_bearish and rsi_overbought:
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
        
        elif position_side == 1:  # Long position
            highest_since_entry = max(highest_since_entry, close[i])
            
            # Trailing stoploss (2*ATR from highest)
            stop_price = highest_since_entry - 2.0 * atr[i]
            
            # Stoploss hit - exit position
            if close[i] < stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            
            # Take profit at 2R, reduce to half
            elif signals[i] == SIZE_FULL:
                risk = entry_price - (highest_since_entry - 2.0 * atr[i])
                profit_target = entry_price + 2.0 * risk
                if close[i] > profit_target:
                    signals[i] = SIZE_HALF
            
            # Exit if trend breaks
            elif not hma_bullish or not daily_bullish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
        
        elif position_side == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, close[i])
            
            # Trailing stoploss (2*ATR from lowest)
            stop_price = lowest_since_entry + 2.0 * atr[i]
            
            # Stoploss hit - exit position
            if close[i] > stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            
            # Take profit at 2R, reduce to half
            elif signals[i] == -SIZE_FULL:
                risk = (lowest_since_entry + 2.0 * atr[i]) - entry_price
                profit_target = entry_price - 2.0 * risk
                if close[i] < profit_target:
                    signals[i] = -SIZE_HALF
            
            # Exit if trend breaks
            elif not hma_bearish or not daily_bearish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
    
    return signals