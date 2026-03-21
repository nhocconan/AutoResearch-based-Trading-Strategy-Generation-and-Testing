#!/usr/bin/env python3
"""
Experiment #220: 4h KAMA Adaptive Trend with Daily HMA Filter and RSI Timing
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - 
moves fast during trends, slow during ranges. This should outperform static EMA/HMA 
in mixed bull/bear/range markets (2021-2024 had all three). Daily HMA provides 
trend bias. RSI(14) for entry timing on pullbacks. Simpler than regime-switching 
to ensure trades actually generate. Position sizing: 0.30 entry, stoploss 2*ATR.
Target: Beat Sharpe=0.499 from current best (12h Supertrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_rsi_timing_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market noise - fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    efficiency_ratio = np.zeros(len(close))
    mask = volatility > 0
    efficiency_ratio[mask] = change[mask] / volatility[mask]
    efficiency_ratio = np.clip(efficiency_ratio, 0, 1)
    
    sc = (efficiency_ratio * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros(len(close))
    kama[period] = close[period]
    for i in range(period+1, len(close)):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # KAMA slope for trend direction
    kama_slope = np.zeros(n)
    for i in range(1, n):
        kama_slope[i] = kama[i] - kama[i-1]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i] and kama_slope[i] > 0
        kama_bearish = close[i] < kama[i] and kama_slope[i] < 0
        
        # RSI signals (looser filters to ensure trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 35 <= rsi[i] <= 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Pullback long: price > KAMA, RSI pullback, daily bullish
        if kama_bullish and daily_bullish:
            if rsi_oversold or (rsi_neutral and rsi[i] < 50):
                new_signal = SIZE_ENTRY
            # Momentum continuation
            elif rsi[i] > 50 and rsi[i] < 65 and close[i] > close[i-1]:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Pullback short: price < KAMA, RSI bounce, daily bearish
        if kama_bearish and daily_bearish:
            if rsi_overbought or (rsi_neutral and rsi[i] > 50):
                new_signal = -SIZE_ENTRY
            # Momentum continuation
            elif rsi[i] < 50 and rsi[i] > 35 and close[i] < close[i-1]:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Trailing stop: 2*ATR from highest
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > stoploss_price:
                stoploss_price = current_stop
            
            # Check stoploss hit
            if close[i] < stoploss_price:
                new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Trailing stop: 2*ATR from lowest
            current_stop = lowest_close + 2.0 * atr[i]
            if stoploss_price == 0.0 or current_stop < stoploss_price:
                stoploss_price = current_stop
            
            # Check stoploss hit
            if close[i] > stoploss_price:
                new_signal = SIZE_EXIT
        
        # === TREND REVERSAL EXIT ===
        # Exit long if KAMA turns bearish
        if position_side > 0 and kama_bearish and daily_bearish:
            new_signal = SIZE_EXIT
        
        # Exit short if KAMA turns bullish
        if position_side < 0 and kama_bullish and daily_bullish:
            new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stoploss_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals