#!/usr/bin/env python3
"""
Experiment #311: 12h KAMA Adaptive Trend + Daily HMA Bias + RSI Pullback + ATR Stops
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market regime - fast in trends,
slow in ranges. Combined with daily HMA for macro bias and RSI pullback entries, this should
work in both bull (2021) and bear/range (2025+) markets. KAMA's efficiency ratio filters
choppy periods automatically. Position size 0.28 with 2.5*ATR stops. Target: Beat Sharpe=0.499.
Timeframe: 12h (required for this experiment), HTF: 1d for trend bias.
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
    Efficiency Ratio (ER) determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Signal = net price change over period
        if i >= period:
            signal = np.abs(close[i] - close[i - period])
        else:
            signal = np.abs(close[i] - close[0])
        
        # Noise = sum of absolute price changes
        if i >= period:
            noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        else:
            noise = np.sum(np.abs(np.diff(close[:i + 1])))
        
        # Efficiency Ratio (0 = noise, 1 = trend)
        if noise > 0:
            er = signal / noise
        else:
            er = 0.0
        
        # Smoothing constant
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_kama_er(close, period=10):
    """Calculate Efficiency Ratio for KAMA (regime filter)."""
    n = len(close)
    er = np.zeros(n)
    
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    return er

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
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    kama_fast = calculate_kama(close, period=5, fast=2, slow=15)  # Faster KAMA for signals
    sma_50 = calculate_sma(close, 50)
    er = calculate_kama_er(close, 10)  # Efficiency ratio for regime
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_kama_fast = np.roll(kama_fast, 1)
    prev_kama_fast[0] = kama_fast[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(sma_50[i]) or np.isnan(er[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # KAMA trend direction
        kama_bullish = kama[i] > prev_kama[i] and close[i] > kama[i]
        kama_bearish = kama[i] < prev_kama[i] and close[i] < kama[i]
        
        # Fast KAMA crossover signals
        kama_cross_long = prev_kama_fast[i] <= prev_kama[i] and kama_fast[i] > kama[i]
        kama_cross_short = prev_kama_fast[i] >= prev_kama[i] and kama_fast[i] < kama[i]
        
        # RSI pullback zones (buy dips in uptrend, sell rallies in downtrend)
        rsi_pullback_long = 35 < rsi[i] < 55 and prev_rsi[i] < 45
        rsi_pullback_short = 45 < rsi[i] < 65 and prev_rsi[i] > 55
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 70
        rsi_momentum_short = rsi[i] < 50 and rsi[i] > 30
        
        # Regime filter (ER > 0.5 = trending, ER < 0.3 = ranging)
        trending_regime = er[i] > 0.4
        ranging_regime = er[i] < 0.35
        
        # Price position vs SMA50
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Daily bullish + KAMA bullish + RSI pullback
        if daily_bullish and kama_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: Daily bullish + KAMA cross long + RSI momentum
        elif daily_bullish and kama_cross_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross long + Above SMA50 + Trending regime
        elif kama_cross_long and above_sma50 and trending_regime:
            new_signal = SIZE_ENTRY
        # Quaternary: Daily bullish + Above SMA50 + RSI 45-60 (simple trend follow)
        elif daily_bullish and above_sma50 and 45 < rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Simple: KAMA bullish + RSI > 50 (catch trends)
        elif kama_bullish and rsi[i] > 50 and above_sma50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Daily bearish + KAMA bearish + RSI pullback
        if daily_bearish and kama_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Daily bearish + KAMA cross short + RSI momentum
        elif daily_bearish and kama_cross_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross short + Below SMA50 + Trending regime
        elif kama_cross_short and below_sma50 and trending_regime:
            new_signal = -SIZE_ENTRY
        # Quaternary: Daily bearish + Below SMA50 + RSI 40-55 (simple trend follow)
        elif daily_bearish and below_sma50 and 40 < rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Simple: KAMA bearish + RSI < 50 (catch trends)
        elif kama_bearish and rsi[i] < 50 and below_sma50:
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