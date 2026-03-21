#!/usr/bin/env python3
"""
Experiment #339: 1h KAMA Adaptive Trend + 4h HMA Filter + RSI Momentum + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - 
smooths in choppy markets, follows closely in trends. Combined with 4h HMA trend 
filter via mtf_data helper, this should capture trends while avoiding whipsaws.
RSI(14) filter ensures momentum confirmation without being too restrictive.
Timeframe: 1h (REQUIRED for this experiment), HTF: 4h for trend bias.
Target: Beat Sharpe=0.499 with 30-60 trades/year, adaptive to market regime.
Key insight: KAMA's adaptive nature should outperform fixed EMAs in mixed markets.
Position sizing: 0.25 entry, 0.125 half (take profit), discrete levels to minimize fees.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_momentum_atr_v1"
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
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |net change| / sum of absolute changes (0 to 1)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    sum_changes = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    sum_changes[:period] = np.nan
    
    er = np.where(sum_changes > 0, net_change / sum_changes, 0.0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
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

def calculate_chop(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market.
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr_sum).rolling(window=period, min_periods=period).sum().values
    
    chop = np.zeros(len(close))
    mask = (highest - lowest) > 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / (highest[mask] - lowest[mask])) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

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
    kama_fast = calculate_kama(close, period=5, fast=2, slow=15)
    chop = calculate_chop(high, low, close, 14)
    
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # KAMA crossover signals (adaptive trend)
        kama_cross_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # KAMA trend state
        kama_bullish = kama_fast[i] > kama[i]
        kama_bearish = kama_fast[i] < kama[i]
        
        # Choppiness regime filter
        is_trending = chop[i] < 50  # Looser than 38.2 to get more trades
        is_ranging = chop[i] > 50
        
        # RSI momentum filter (LOOSE to ensure trades)
        rsi_ok_long = rsi[i] > 40  # Not oversold
        rsi_ok_short = rsi[i] < 60  # Not overbought
        rsi_strong_long = rsi[i] > 50
        rsi_strong_short = rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross + 4h bullish + RSI ok
        if kama_cross_long and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + 4h bullish + RSI strong (no cross needed)
        elif kama_bullish and trend_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross in trending market (momentum only)
        elif kama_cross_long and is_trending and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Quaternary: Simple KAMA cross with RSI filter (ensure trades)
        elif kama_cross_long and rsi[i] > 45 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross + 4h bearish + RSI ok
        if kama_cross_short and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + 4h bearish + RSI strong (no cross needed)
        elif kama_bearish and trend_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross in trending market (momentum only)
        elif kama_cross_short and is_trending and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Quaternary: Simple KAMA cross with RSI filter (ensure trades)
        elif kama_cross_short and rsi[i] < 55 and rsi[i] > 30:
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