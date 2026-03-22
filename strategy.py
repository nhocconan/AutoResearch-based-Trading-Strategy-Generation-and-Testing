#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA + Fisher Transform + Choppiness Index + Daily Bias
Hypothesis: 4h timeframe balances noise reduction with trade frequency. KAMA adapts
to volatility better than EMA/HMA. Fisher Transform catches reversals in bear markets
(where simple trend strategies fail). Choppiness Index filters range vs trend regimes.
Daily KAMA provides HTF bias alignment. Multiple entry paths ensure >=10 trades.
Conservative sizing (0.25) with 2.5*ATR stoploss controls drawdown.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_daily_bias_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency.
    ER (Efficiency Ratio) measures trend vs noise.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    sc[er_period:] = (er[er_period:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Better reversal detection than RSI in bear markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            continue
        
        # Normalize price
        x = (2.0 * (close[i] - ll) / (hh - ll)) - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies range-bound vs trending markets.
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow).
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > kama_1d_aligned[i]
        daily_bearish = close[i] < kama_1d_aligned[i]
        
        # 4h KAMA trend
        kama_4h_bullish = close[i] > kama_4h[i]
        kama_4h_bearish = close[i] < kama_4h[i]
        kama_rising = kama_4h[i] > kama_4h[i-1] if i > 0 else False
        kama_falling = kama_4h[i] < kama_4h[i-1] if i > 0 else False
        
        # Fast KAMA crossover
        fast_above_slow = kama_4h_fast[i] > kama_4h[i]
        fast_below_slow = kama_4h_fast[i] < kama_4h[i]
        
        # Fisher Transform signals
        fisher_bullish_cross = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5 if i > 0 else False
        fisher_bearish_cross = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5 if i > 0 else False
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # Choppiness Index regime
        chop_range = chop[i] > 55.0  # Range-bound market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Daily bullish + 4h KAMA bullish + Fast KAMA crossover + Trend regime
        if daily_bullish and kama_4h_bullish and fast_above_slow and chop_trend:
            new_signal = SIZE_ENTRY
        
        # Path 2: Fisher bullish cross + Daily not bearish + RSI pullback
        elif fisher_bullish_cross and not daily_bearish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 3: Range market + Fisher extreme low + RSI oversold (mean reversion)
        elif chop_range and fisher_extreme_low and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 4: Daily bullish + KAMA rising + RSI neutral + Fast KAMA rising
        elif daily_bullish and kama_rising and rsi[i] > 45 and rsi[i] < 55 and kama_4h_fast[i] > kama_4h_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 5: Fisher bullish cross + KAMA 4h bullish + ADX-like filter (chop < 50)
        elif fisher_bullish_cross and kama_4h_bullish and chop[i] < 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Daily bearish + 4h KAMA bearish + Fast KAMA crossover + Trend regime
        if daily_bearish and kama_4h_bearish and fast_below_slow and chop_trend:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Fisher bearish cross + Daily not bullish + RSI pullback
        elif fisher_bearish_cross and not daily_bullish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Range market + Fisher extreme high + RSI overbought (mean reversion)
        elif chop_range and fisher_extreme_high and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Daily bearish + KAMA falling + RSI neutral + Fast KAMA falling
        elif daily_bearish and kama_falling and rsi[i] > 45 and rsi[i] < 55 and kama_4h_fast[i] < kama_4h_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Fisher bearish cross + KAMA 4h bearish + ADX-like filter (chop < 50)
        elif fisher_bearish_cross and kama_4h_bearish and chop[i] < 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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