#!/usr/bin/env python3
"""
Experiment #232: 4h KAMA Adaptive Trend with Daily HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility,
reducing whipsaw during choppy periods while capturing trends efficiently.
On 4h timeframe, KAMA(10) crossover with KAMA(40) provides entry signals.
Daily HMA(21) provides trend bias (only long when price > 1d HMA, short when <).
This is simpler than multi-filter regimes and should generate consistent trades
across BTC/ETH/SOL. Position sizing: 0.25 entry, 0.125 half at 2R profit.
Stoploss: 2.5*ATR trailing stop. Target: Beat Sharpe=0.499 from current best.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_atr_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast during trends, slow during chop.
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=period, min_periods=period).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] ** 2 * (close_s.iloc[i] - kama.iloc[i-1])
    
    return kama.values

def calculate_momentum(close, period=10):
    """Calculate simple momentum (rate of change)."""
    mom = close - np.roll(close, period)
    mom[:period] = 0.0
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    momentum = calculate_momentum(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA crossover signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover detection (fast crosses above/below slow)
        kama_cross_long = (kama_fast[i] > kama_slow[i] and 
                          kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_short = (kama_fast[i] < kama_slow[i] and 
                           kama_fast[i-1] >= kama_slow[i-1])
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0
        mom_negative = momentum[i] < 0
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA crossover long with daily trend confirmation
        if kama_cross_long:
            if daily_bullish and mom_positive:
                new_signal = SIZE_ENTRY
            elif daily_bullish:
                new_signal = SIZE_ENTRY * 0.8
        
        # KAMA trend continuation (already bullish, pullback entry)
        elif kama_bullish and daily_bullish:
            # Enter on pullback when momentum turns positive again
            if momentum[i-1] <= 0 and mom_positive:
                new_signal = SIZE_ENTRY * 0.6
        
        # === SHORT ENTRY ===
        # KAMA crossover short with daily trend confirmation
        if kama_cross_short:
            if daily_bearish and mom_negative:
                new_signal = -SIZE_ENTRY
            elif daily_bearish:
                new_signal = -SIZE_ENTRY * 0.8
        
        # KAMA trend continuation (already bearish, pullback entry)
        elif kama_bearish and daily_bearish:
            # Enter on pullback when momentum turns negative again
            if momentum[i-1] >= 0 and mom_negative:
                new_signal = -SIZE_ENTRY * 0.6
        
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
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
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
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals