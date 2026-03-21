#!/usr/bin/env python3
"""
Hypothesis: 15m entries with 4h trend filter + Supertrend + RSI pullback
- 4h HMA determines primary trend direction (HTF filter)
- 15m Supertrend confirms momentum and provides trailing stop
- RSI pullback entries in trend direction (buy dips in uptrend, sell rips in downtrend)
- ATR-based stoploss (2*ATR) exits positions when trend breaks
- Discrete position sizing (0.0, ±0.25, ±0.30) to minimize fee churn
- Must generate sufficient trades across BTC/ETH/SOL
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing method."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction and trailing stop."""
    n = len(close)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i-1] > supertrend[i-1]:
            # Previously bullish
            if close[i] > lower_band[i]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously bearish
            if close[i] < upper_band[i]:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smooth trend following."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Calculate 15m indicators
    atr_15m = calculate_atr(high, low, close, 14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, atr_15m, 3.0)
    rsi_15m = calculate_rsi(close, 14)
    hma_15m = calculate_hma(close, 21)
    
    # Load 4h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1) for completed bars
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_EXIT = 0.0
    SIZE_HALF = 0.12
    
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # 4h trend filter (HTF)
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 15m supertrend direction
        st_trend = int(st_direction_15m[i])
        
        # 15m HMA confirmation
        hma_15m_trend = 1 if hma_15m[i] > hma_15m[i-5] else -1
        
        # RSI pullback logic
        rsi_signal = 0
        if hma_4h_trend > 0:  # 4h uptrend - look for long entries
            if rsi_15m[i] < 45:  # RSI pullback (oversold in uptrend)
                rsi_signal = 1
            elif rsi_15m[i] > 70:  # Overbought - reduce position
                rsi_signal = -1
        elif hma_4h_trend < 0:  # 4h downtrend - look for short entries
            if rsi_15m[i] > 55:  # RSI pullback (overbought in downtrend)
                rsi_signal = -1
            elif rsi_15m[i] < 30:  # Oversold - reduce position
                rsi_signal = 1
        
        # Stoploss check (2*ATR from entry)
        if position_side != 0:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stoploss_price = entry_price - 2.0 * atr_15m[i]
                trailing_stop = highest_since_entry - 2.0 * atr_15m[i]
                stop_price = max(stoploss_price, trailing_stop)
                
                if close[i] < stop_price:
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    entry_price = 0.0
                    continue
            else:  # short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stoploss_price = entry_price + 2.0 * atr_15m[i]
                trailing_stop = lowest_since_entry + 2.0 * atr_15m[i]
                stop_price = min(stoploss_price, trailing_stop)
                
                if close[i] > stop_price:
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    entry_price = 0.0
                    continue
        
        # Take profit at 2R (reduce to half position)
        if position_side != 0:
            if position_side > 0:
                profit_r = (close[i] - entry_price) / atr_15m[i]
                if profit_r >= 2.0 and signals[i-1] >= SIZE_ENTRY:
                    signals[i] = SIZE_HALF
                    continue
            else:
                profit_r = (entry_price - close[i]) / atr_15m[i]
                if profit_r >= 2.0 and signals[i-1] <= -SIZE_ENTRY:
                    signals[i] = -SIZE_HALF
                    continue
        
        # Entry logic - all filters must agree
        if position_side == 0:
            # Long entry: 4h uptrend + Supertrend bullish + RSI pullback
            if hma_4h_trend > 0 and st_trend > 0 and rsi_signal > 0:
                signals[i] = SIZE_ENTRY
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            # Short entry: 4h downtrend + Supertrend bearish + RSI pullback
            elif hma_4h_trend < 0 and st_trend < 0 and rsi_signal < 0:
                signals[i] = -SIZE_ENTRY
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
            else:
                signals[i] = SIZE_EXIT
        else:
            # Hold position or exit on trend reversal
            if position_side > 0:
                if st_trend < 0 or hma_4h_trend < 0:
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    entry_price = 0.0
                else:
                    signals[i] = signals[i] if signals[i] != 0 else SIZE_ENTRY
            else:
                if st_trend > 0 or hma_4h_trend > 0:
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    entry_price = 0.0
                else:
                    signals[i] = signals[i] if signals[i] != 0 else -SIZE_ENTRY
    
    return signals