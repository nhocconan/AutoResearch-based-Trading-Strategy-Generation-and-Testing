#!/usr/bin/env python3
"""
Experiment #365: 12h KAMA Adaptive Trend + Daily HMA + RSI Momentum + Volatility Filter
Hypothesis: KAMA adapts to market efficiency better than HMA/EMA, reducing whipsaws in range markets.
12h timeframe captures medium-term trends with fewer false signals than 4h. Daily HMA provides soft trend bias.
RSI(14) with moderate thresholds (35-65) ensures sufficient trade frequency without overtrading.
ATR-based volatility filter avoids entries during extreme moves. Stoploss at 2.5*ATR protects capital.
Building on #359 (Sharpe=0.061) by replacing Donchian with KAMA for smoother trend following.
Target: Beat Sharpe=0.499 with 40-80 trades total, DD < -30%.
Key insight: KAMA's efficiency ratio adapts position sizing implicitly - faster in trends, slower in ranges.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_vol_filter_atr_v1"
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

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er = price_change / volatility
        else:
            er = 0
        
        # Calculate smoothing constant
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:period] = close[:period]
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
    kama = calculate_kama(close, 14, fast=2, slow=30)
    
    # KAMA slope (momentum)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = kama[i] - kama[i - 5]
    
    # Volatility filter (ATR ratio)
    atr_ratio = atr / close
    atr_median = np.nanmedian(atr_ratio[100:])
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (SOFT filter - boosts confidence, not required)
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # KAMA trend signals
        kama_bullish = kama_slope[i] > 0 and close[i] > kama[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama[i]
        
        # RSI momentum filter (MODERATE thresholds for trade frequency)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        rsi_strong_long = rsi[i] > 45 and rsi[i] < 70
        rsi_strong_short = rsi[i] > 30 and rsi[i] < 55
        
        # Volatility filter (avoid extreme volatility)
        vol_ok = atr_ratio[i] < atr_median * 2.5  # Not in extreme volatility
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA bullish + RSI ok + Daily bullish + Vol ok
        if kama_bullish and rsi_ok_long and daily_bullish and vol_ok:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + RSI strong + Vol ok (daily neutral ok)
        elif kama_bullish and rsi_strong_long and vol_ok:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA crossover (close crosses above KAMA) + RSI ok
        elif close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA bullish alone (ensures minimum trade frequency)
        elif kama_bullish and rsi[i] > 40 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA bearish + RSI ok + Daily bearish + Vol ok
        if kama_bearish and rsi_ok_short and daily_bearish and vol_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + RSI strong + Vol ok (daily neutral ok)
        elif kama_bearish and rsi_strong_short and vol_ok:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA crossover (close crosses below KAMA) + RSI ok
        elif close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA bearish alone (ensures minimum trade frequency)
        elif kama_bearish and rsi[i] > 30 and rsi[i] < 60:
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