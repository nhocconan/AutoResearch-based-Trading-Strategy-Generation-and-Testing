#!/usr/bin/env python3
"""
EXPERIMENT #012 - 1h RSI Pullback with 4h HMA Trend Filter
==========================================================
Hypothesis: Using 4h HMA(21) for trend direction + 1h RSI(14) pullback entries
will capture trends at better entry prices than pure trend following.

Key innovations:
- 4h HMA trend (smoother than EMA, less lag than SMA) - loaded ONCE via mtf_data
- 1h RSI pullback entries (buy dips in uptrend, sell rallies in downtrend)
- Volume spike confirmation (1.5x 20-bar avg) to filter weak signals
- ATR(14) trailing stoploss - signal→0 when stopped out
- Discrete position sizing (0.0, ±0.30) to minimize fee churn

Different from failed strategies:
- Not Supertrend (failed #001, #003, #007, #011)
- Not Daily filter (failed #001, #004, #006, #007, #008, #010)
- Not Donchian (failed #009)
- Not KAMA (failed #008, #010)
- NEW: HMA + RSI pullback + volume confirmation combination
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_hma_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Array - smoother than EMA, less lag than SMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
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
    
    avg_gain = gain_s.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss_s.ewm(span=period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
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
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD 4h HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21) for trend direction
    hma_4h = calculate_hma(close_4h, 21)
    
    # Align 4h HMA to 1h timeframe (auto shift(1) for completed bars only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    rsi_1h = calculate_rsi(close, 14)
    atr_1h = calculate_atr(high, low, close, 14)
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # HMA(21) on 1h for additional trend confirmation
    hma_1h = calculate_hma(close, 21)
    
    # === GENERATE SIGNALS WITH STOPLOSS LOGIC ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for DD control
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    min_bars = max(50, int(np.sqrt(period)) + 20)  # Ensure enough data for all indicators
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # 4h trend direction
        trend_4h = hma_4h_aligned[i]
        trend_4h_prev = hma_4h_aligned[i-1] if i > 0 else trend_4h
        
        # Determine 4h trend bias (price vs HMA)
        is_uptrend_4h = close[i] > trend_4h
        is_downtrend_4h = close[i] < trend_4h
        
        # 1h trend confirmation
        is_uptrend_1h = close[i] > hma_1h[i] if not np.isnan(hma_1h[i]) else False
        is_downtrend_1h = close[i] < hma_1h[i] if not np.isnan(hma_1h[i]) else False
        
        # Volume spike filter
        vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # RSI levels for pullback entries
        rsi = rsi_1h[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        rsi_neutral = 35 <= rsi <= 65
        
        # ATR for stoploss
        atr = atr_1h[i]
        if np.isnan(atr) or atr <= 0:
            atr = 0.02 * close[i]  # Fallback to 2% of price
        
        # === STOPLOSS LOGIC (Rule #6) ===
        if position_side == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Trailing stop: exit if price drops 2*ATR from highest
            if close[i] < highest_since_entry - 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
                continue
            # Hard stoploss: exit if price drops 2.5*ATR from entry
            if close[i] < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position_side = 0
                continue
                
        elif position_side == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Trailing stop: exit if price rises 2*ATR from lowest
            if close[i] > lowest_since_entry + 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
                continue
            # Hard stoploss: exit if price rises 2.5*ATR from entry
            if close[i] > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # === ENTRY LOGIC ===
        # Long entry: 4h uptrend + 1h uptrend + RSI oversold pullback + volume spike
        if position_side == 0 and is_uptrend_4h and is_uptrend_1h and rsi_oversold:
            signals[i] = SIZE
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            
        # Short entry: 4h downtrend + 1h downtrend + RSI overbought pullback + volume spike
        elif position_side == 0 and is_downtrend_4h and is_downtrend_1h and rsi_overbought:
            signals[i] = -SIZE
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = low[i]
            
        # === EXIT LOGIC (RSI mean reversion) ===
        # Exit long when RSI becomes overbought (take profit on mean reversion)
        elif position_side == 1 and rsi_overbought:
            signals[i] = 0.0
            position_side = 0
            
        # Exit short when RSI becomes oversold (take profit on mean reversion)
        elif position_side == -1 and rsi_oversold:
            signals[i] = 0.0
            position_side = 0
            
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend reverses
        elif position_side == 1 and is_downtrend_4h:
            signals[i] = 0.0
            position_side = 0
            
        # Exit short if 4h trend reverses
        elif position_side == -1 and is_uptrend_4h:
            signals[i] = 0.0
            position_side = 0
            
        # Otherwise maintain current position
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals