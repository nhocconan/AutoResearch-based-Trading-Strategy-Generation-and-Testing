#!/usr/bin/env python3
"""
Experiment #233: 12h KAMA Adaptive Trend with Daily HMA Filter and RSI Pullback
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - 
fast during trends, slow during ranges. This should outperform fixed-period EMAs 
in crypto's varying volatility regimes. Combined with daily HMA trend filter 
(only trade in direction of 1d trend) and RSI pullback entries (enter on dips 
in uptrend, rallies in downtrend). This avoids breakout false signals that failed 
in exp #224. Position sizing: 0.30 entry, 0.15 at 2R profit. Stoploss: 2.5*ATR.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_pullback_atr_v1"
timeframe = "12h"
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
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - fast in trends, slow in ranges.
    ER (Efficiency Ratio) measures trend strength.
    SC (Smoothing Constant) adjusts based on ER.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Calculate Efficiency Ratio (ER)
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        er = signal / noise if noise > 0 else 0.0
        
        # Calculate Smoothing Constant (SC)
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
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
        
        # RSI pullback levels
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_neutral = 40 < rsi[i] < 60
        
        # KAMA crossover signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover detection (cross happened this bar)
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # Price position relative to KAMA
        above_kama = close[i] > kama_slow[i]
        below_kama = close[i] < kama_slow[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA bullish crossover with daily trend confirmation
        if kama_cross_long:
            if daily_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            elif daily_bullish and rsi_neutral:
                new_signal = SIZE_ENTRY
        
        # Pullback entry in uptrend (price dips to KAMA then bounces)
        elif daily_bullish and kama_bullish:
            # Price was below KAMA, now above (pullback complete)
            if close[i-1] < kama_slow[i-1] and close[i] > kama_slow[i]:
                if rsi_oversold or rsi_neutral:
                    new_signal = SIZE_ENTRY
            # Price above KAMA, RSI pulled back
            elif above_kama and rsi_oversold:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA bearish crossover with daily trend confirmation
        if kama_cross_short:
            if daily_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
            elif daily_bearish and rsi_neutral:
                new_signal = -SIZE_ENTRY
        
        # Pullback entry in downtrend (price rallies to KAMA then drops)
        elif daily_bearish and kama_bearish:
            # Price was above KAMA, now below (pullback complete)
            if close[i-1] > kama_slow[i-1] and close[i] < kama_slow[i]:
                if rsi_overbought or rsi_neutral:
                    new_signal = -SIZE_ENTRY
            # Price below KAMA, RSI rallied
            elif below_kama and rsi_overbought:
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