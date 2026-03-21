#!/usr/bin/env python3
"""
Hypothesis: 15m Supertrend + 4h HMA trend filter + RSI pullback entries
- 4h HMA determines primary trend direction (HTF filter)
- 15m Supertrend provides entry timing
- RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
- ATR(14) stoploss at 2*ATR from entry
- Discrete position sizes: 0.0, ±0.25, ±0.35
- This should reduce false signals vs pure 15m strategies while generating enough trades
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_4h_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    supertrend[0] = upper[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1] if direction[i-1] == 1 else lower[i])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1] if direction[i-1] == -1 else upper[i])
            direction[i] = -1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = np.zeros(len(close))
    avg_l = np.zeros(len(close))
    
    avg_g[0] = np.mean(gain[:period]) if len(gain) >= period else np.mean(gain)
    avg_l[0] = np.mean(loss[:period]) if len(loss) >= period else np.mean(loss)
    
    for i in range(1, len(close)):
        avg_g[i] = (avg_g[i-1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_15m = calculate_atr(high, low, close, 14)
    supertrend_15m, st_direction = calculate_supertrend(high, low, close, atr_15m, 3.0)
    rsi_15m = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # HTF trend filter: 4h HMA slope
        hma_trend = 1 if hma_4h_aligned[i] > hma_4h_aligned[i-1] else -1
        
        # 15m Supertrend direction
        st_trend = st_direction[i]
        
        # RSI levels
        rsi = rsi_15m[i]
        atr = atr_15m[i]
        
        # Entry logic: HTF trend + ST trend + RSI pullback
        if hma_trend > 0 and st_trend > 0:  # Uptrend
            if rsi < 45 and position_side != 1:  # Pullback entry
                signals[i] = SIZE_ENTRY
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
        elif hma_trend < 0 and st_trend < 0:  # Downtrend
            if rsi > 55 and position_side != -1:  # Rally entry
                signals[i] = -SIZE_ENTRY
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
        else:
            # Hold position if already in one, but check stoploss
            signals[i] = signals[i-1] if i > 0 else 0.0
        
        # Stoploss and take profit logic
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, close[i])
            # Stoploss: 2*ATR below entry
            if close[i] < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
            # Take profit: reduce to half at 2R, trail stop
            elif close[i] > entry_price + 2.0 * atr:
                if signals[i] == SIZE_ENTRY:
                    signals[i] = SIZE_HALF
                # Trail stop: exit if price drops 1*ATR from highest
                if close[i] < highest_since_entry - 1.0 * atr:
                    signals[i] = 0.0
                    position_side = 0
                    
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, close[i])
            # Stoploss: 2*ATR above entry
            if close[i] > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position_side = 0
            # Take profit: reduce to half at 2R, trail stop
            elif close[i] < entry_price - 2.0 * atr:
                if signals[i] == -SIZE_ENTRY:
                    signals[i] = -SIZE_HALF
                # Trail stop: exit if price rises 1*ATR from lowest
                if close[i] > lowest_since_entry + 1.0 * atr:
                    signals[i] = 0.0
                    position_side = 0
    
    return signals