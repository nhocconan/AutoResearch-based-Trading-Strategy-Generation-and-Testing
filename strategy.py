#!/usr/bin/env python3
"""
Hypothesis: 15m Supertrend + 4h HMA trend filter + RSI pullback entries
- 4h HMA(21) determines bull/bear regime (HTF trend bias)
- 15m Supertrend(10,3) generates entry signals in trend direction
- 15m RSI(14) confirms pullback entries (not chasing)
- ATR(14) stoploss exits when price moves 2.5*ATR against position
- Discrete sizing: 0.0, ±0.15, ±0.30 to minimize fee churn
- Target: 30-80 trades/year, works on BTC/ETH/SOL with Sharpe>0 each
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator values."""
    n = len(close)
    atr = np.zeros(n)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = basic_upper[0]
    
    for i in range(1, n):
        # Final upper/lower calculation
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Supertrend direction
        if supertrend[i-1] == final_upper[i-1]:
            if close[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
        else:
            if close[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
    
    # Direction: 1 = bullish (price above supertrend), -1 = bearish
    direction = np.where(close > supertrend, 1.0, -1.0)
    
    return supertrend, direction, atr

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    close_4h_raw = df_4h['close'].values
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h_raw)
    
    # === CALCULATE 15m INDICATORS (vectorized before loop) ===
    supertrend, st_direction, atr = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    
    # 15m HMA for additional trend confirmation
    hma_15m = calculate_hma(close, 21)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Track position for stoploss
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    
    for i in range(50, n):
        # HTF trend bias from 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        close_4h_val = close_4h_aligned[i]
        
        # 4h trend: price above HMA = bullish bias
        htf_bullish = close_4h_val > hma_4h_val if not np.isnan(hma_4h_val) else False
        htf_bearish = close_4h_val < hma_4h_val if not np.isnan(hma_4h_val) else False
        
        # 15m trend from Supertrend
        st_bullish = st_direction[i] == 1.0
        st_bearish = st_direction[i] == -1.0
        
        # 15m HMA confirmation
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        
        # RSI pullback filter (avoid chasing)
        rsi_neutral = 35 < rsi[i] < 65
        rsi_bull_pullback = 40 < rsi[i] < 55  # pullback in uptrend
        rsi_bear_pullback = 45 < rsi[i] < 60  # pullback in downtrend
        
        # === ENTRY LOGIC ===
        # Long entry: HTF bullish + 15m Supertrend bullish + RSI pullback
        if htf_bullish and st_bullish and hma_15m_bullish:
            if rsi_bull_pullback or (rsi[i] < 50 and st_direction[i-1] == 1.0):
                if position_side != 1:
                    signals[i] = SIZE_FULL
                    entry_price = close[i]
                    position_side = 1
                else:
                    signals[i] = SIZE_FULL
            elif position_side == 1:
                signals[i] = SIZE_FULL
            else:
                signals[i] = 0.0
        
        # Short entry: HTF bearish + 15m Supertrend bearish + RSI pullback
        elif htf_bearish and st_bearish and hma_15m_bearish:
            if rsi_bear_pullback or (rsi[i] > 50 and st_direction[i-1] == -1.0):
                if position_side != -1:
                    signals[i] = -SIZE_FULL
                    entry_price = close[i]
                    position_side = -1
                else:
                    signals[i] = -SIZE_FULL
            elif position_side == -1:
                signals[i] = -SIZE_FULL
            else:
                signals[i] = 0.0
        
        # Flat when HTF and 15m disagree (choppy market)
        else:
            if position_side == 1:
                signals[i] = SIZE_HALF  # reduce but don't exit yet
            elif position_side == -1:
                signals[i] = -SIZE_HALF
            else:
                signals[i] = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side == 1 and atr[i] > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
        
        if position_side == -1 and atr[i] > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if position_side == 1 and atr[i] > 0:
            take_profit = entry_price + 2.0 * 2.5 * atr[i]  # 2R profit
            if close[i] > take_profit and signals[i] == SIZE_FULL:
                signals[i] = SIZE_HALF
        
        if position_side == -1 and atr[i] > 0:
            take_profit = entry_price - 2.0 * 2.5 * atr[i]  # 2R profit
            if close[i] < take_profit and signals[i] == -SIZE_FULL:
                signals[i] = -SIZE_HALF
    
    return signals