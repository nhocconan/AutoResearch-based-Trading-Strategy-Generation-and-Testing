#!/usr/bin/env python3
"""
Experiment #237: 1h KAMA Adaptive Trend with 4h HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - 
fast in trends, slow in chop. Combined with 4h HMA for trend direction, this should 
work better than static EMAs. Simple RSI filter (not extreme) avoids bad entries. 
Fewer filters = more trades. Position sizing: 0.25 entry, stop at 2.5*ATR.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market noise - fast during trends, slow during chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = absolute price change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.abs(close[:period] - close[0])
    
    # Volatility = sum of absolute price changes over period
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    volatility[:period] = np.abs(close[:period] - np.roll(close[:period], 1)).sum()
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period-1] = close[period-1]
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # KAMA slope for momentum confirmation
    kama_slope = np.diff(kama, prepend=kama[0])
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (4h HMA)
        hma_bullish = close[i] > hma_4h_aligned[i]
        hma_bearish = close[i] < hma_4h_aligned[i]
        
        # RSI filter (avoid extremes but not too strict)
        rsi_ok_long = rsi[i] > 35  # not oversold
        rsi_ok_short = rsi[i] < 65  # not overbought
        
        # KAMA crossover signals
        kama_cross_long = close[i] > kama[i] and close[i-1] <= kama[i-1]
        kama_cross_short = close[i] < kama[i] and close[i-1] >= kama[i-1]
        
        # KAMA momentum (price above KAMA + KAMA sloping up)
        kama_momentum_long = close[i] > kama[i] and kama_slope[i] > 0
        kama_momentum_short = close[i] < kama[i] and kama_slope[i] < 0
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA crossover with HTF trend confirmation
        if kama_cross_long:
            if hma_bullish and rsi_ok_long:
                new_signal = SIZE_ENTRY
            elif not hma_bearish and rsi_ok_long:  # neutral or bullish 4h
                new_signal = SIZE_ENTRY
        
        # KAMA momentum continuation
        elif kama_momentum_long and hma_bullish and rsi_ok_long:
            # Enter on pullback to KAMA
            if close[i-1] < kama[i-1] and close[i] > kama[i]:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA crossover with HTF trend confirmation
        if kama_cross_short:
            if hma_bearish and rsi_ok_short:
                new_signal = -SIZE_ENTRY
            elif not hma_bullish and rsi_ok_short:  # neutral or bearish 4h
                new_signal = -SIZE_ENTRY
        
        # KAMA momentum continuation
        elif kama_momentum_short and hma_bearish and rsi_ok_short:
            # Enter on pullback to KAMA
            if close[i-1] > kama[i-1] and close[i] < kama[i]:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = SIZE_EXIT
            
            # KAMA reversal stop
            elif close[i] < kama[i] and kama_slope[i] < 0:
                new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = SIZE_EXIT
            
            # KAMA reversal stop
            elif close[i] > kama[i] and kama_slope[i] > 0:
                new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.5 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.5 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals