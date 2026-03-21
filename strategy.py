#!/usr/bin/env python3
"""
Experiment #246: 1d KAMA Adaptive Trend + Weekly HMA + ROC Momentum
Hypothesis: Daily KAMA (Kaufman Adaptive Moving Average) adapts to volatility changes
better than fixed EMA/HMA, reducing whipsaws in ranging markets. Weekly HMA provides
macro trend bias. ROC(10) momentum filter ensures we only enter when momentum confirms.
Simple 2-condition entry (KAMA crossover + weekly trend) to ensure sufficient trades.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 with fewer whipsaws in 2025 bear market.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_roc_momentum_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = absolute price change over er_period
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    # Volatility = sum of absolute single-period changes
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    # Efficiency Ratio (ER) = change / volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility != 0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
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

def calculate_roc(close, period=10):
    """Calculate Rate of Change momentum indicator."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.values

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    roc = calculate_roc(close, 10)
    rsi = calculate_rsi(close, 14)
    
    # Track previous values for crossover detection
    prev_kama = np.roll(kama, 1)
    prev_kama_fast = np.roll(kama_fast, 1)
    prev_kama[0] = kama[0]
    prev_kama_fast[0] = kama_fast[0]
    
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
        # Weekly trend filter (macro bias)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # KAMA crossover signals (adaptive trend)
        kama_cross_up = prev_kama[i] <= kama[i] and kama_fast[i] > kama[i]
        kama_cross_down = prev_kama[i] >= kama[i] and kama_fast[i] < kama[i]
        
        # KAMA slope (trend direction)
        kama_slope_up = kama[i] > prev_kama[i]
        kama_slope_down = kama[i] < prev_kama[i]
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # ROC momentum filter
        roc_strong_long = roc[i] > 2.0
        roc_strong_short = roc[i] < -2.0
        roc_positive = roc[i] > 0
        roc_negative = roc[i] < 0
        
        # RSI filter (not extreme)
        rsi_ok_long = 35 < rsi[i] < 75
        rsi_ok_short = 25 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA fast crosses above KAMA with weekly trend
        if kama_cross_up:
            if weekly_bullish and roc_positive and rsi_ok_long:
                new_signal = SIZE_ENTRY
            elif price_above_kama and roc_strong_long:
                new_signal = SIZE_ENTRY
        
        # Price crosses above KAMA with momentum
        elif price_above_kama and prev_kama[i] >= close[i]:
            if weekly_bullish and kama_slope_up:
                new_signal = SIZE_ENTRY
        
        # Pullback to KAMA in uptrend
        elif price_above_kama and weekly_bullish:
            if close[i-1] < kama[i-1] and close[i] > kama[i]:
                if roc_positive or kama_slope_up:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA fast crosses below KAMA with weekly trend
        if kama_cross_down:
            if weekly_bearish and roc_negative and rsi_ok_short:
                new_signal = -SIZE_ENTRY
            elif price_below_kama and roc_strong_short:
                new_signal = -SIZE_ENTRY
        
        # Price crosses below KAMA with momentum
        elif price_below_kama and prev_kama[i] <= close[i]:
            if weekly_bearish and kama_slope_down:
                new_signal = -SIZE_ENTRY
        
        # Pullback to KAMA in downtrend
        elif price_below_kama and weekly_bearish:
            if close[i-1] > kama[i-1] and close[i] < kama[i]:
                if roc_negative or kama_slope_down:
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