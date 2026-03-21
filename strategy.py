#!/usr/bin/env python3
"""
Experiment #366: 1d KAMA Adaptive Trend + Weekly HMA Macro + RSI Momentum + ATR Stop
Hypothesis: Daily timeframe captures major trends with minimal noise. KAMA adapts to volatility
(regime changes), Weekly HMA provides macro bias filter, RSI(14) with moderate thresholds (30-70)
ensures sufficient trade frequency on daily bars. ATR(14) stoploss at 2.5x protects capital.
Timeframe: 1d (REQUIRED), HTF: 1w for macro trend via mtf_data helper.
Target: Beat Sharpe=0.499 with 15-40 trades total across train+test.
Key insight: 1d has fewer whipsaws than lower TFs, KAMA adapts better than EMA in crypto volatility.
Position sizing: 0.30 entry, 0.15 half-exit at 2R profit, discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_rsi_momentum_atr_v1"
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
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - fast in trends, slow in chop.
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, er_period))
    net_change[:er_period] = np.abs(close[:er_period] - close[0])
    
    abs_changes = np.abs(close - np.roll(close, 1))
    abs_changes[0] = abs_changes[1] if len(abs_changes) > 1 else 0.0
    
    sum_abs_changes = pd.Series(abs_changes).rolling(window=er_period, min_periods=er_period).sum().values
    sum_abs_changes[:er_period] = np.sum(abs_changes[:er_period])
    
    er = np.where(sum_abs_changes > 0, net_change / sum_abs_changes, 0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    
    # KAMA slope (trend direction)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = kama[i] - kama[i-5]
    
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
    
    for i in range(50, n):  # Start after 50 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # KAMA trend direction
        kama_bullish = kama_slope[i] > 0
        kama_bearish = kama_slope[i] < 0
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum (moderate thresholds for trade frequency on 1d)
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 65
        rsi_strong_long = rsi[i] > 45 and rsi[i] < 75
        rsi_strong_short = rsi[i] > 25 and rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Weekly bullish + KAMA bullish + Price above KAMA + RSI neutral
        if weekly_bullish and kama_bullish and price_above_kama and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + KAMA bullish + RSI strong (price filter relaxed)
        elif weekly_bullish and kama_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA bullish + Price above KAMA + RSI ok (weekly neutral ok)
        elif kama_bullish and price_above_kama and rsi[i] > 40 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA slope strongly positive + RSI ok (ensures minimum trades)
        elif kama_slope[i] > atr[i] * 0.5 and rsi[i] > 35 and rsi[i] < 80:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Weekly bearish + KAMA bearish + Price below KAMA + RSI neutral
        if weekly_bearish and kama_bearish and price_below_kama and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + KAMA bearish + RSI strong (price filter relaxed)
        elif weekly_bearish and kama_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA bearish + Price below KAMA + RSI ok (weekly neutral ok)
        elif kama_bearish and price_below_kama and rsi[i] > 25 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA slope strongly negative + RSI ok (ensures minimum trades)
        elif kama_slope[i] < -atr[i] * 0.5 and rsi[i] > 20 and rsi[i] < 65:
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