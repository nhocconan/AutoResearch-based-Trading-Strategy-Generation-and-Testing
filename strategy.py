#!/usr/bin/env python3
"""
Experiment #364: 4h Fisher Transform + KAMA Trend + Daily HMA + Choppiness Regime + ATR Stop
Hypothesis: 4h timeframe with Fisher Transform catches reversals better than RSI in bear/range markets.
KAMA adapts to volatility (slower in chop, faster in trends). Daily HMA provides macro bias (proven in #359).
Choppiness Index filters regime: CHOP>61.8 = range (use mean reversion), CHOP<38.2 = trend (use breakout).
This combines reversal detection (Fisher) with adaptive trend (KAMA) and regime filter (CHOP).
Target: Beat Sharpe=0.499 with 20-50 trades per symbol, DD < -30%.
Key insight: Fisher Transform normalizes price to Gaussian distribution, better for extreme reversals.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_daily_hma_chop_regime_atr_v1"
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
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            change = np.abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            er = change / volatility if volatility > 0 else 0
        else:
            er = 0
        
        # Smoothing constant
        sc = (er * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        # Normalize price
        range_hl = highest - lowest
        if range_hl > 0:
            normalized = 0.66 * ((close[i] - lowest) / range_hl - 0.5) + 0.67 * (0 if i == period else normalized_prev)
        else:
            normalized = 0
        
        # Clamp to prevent division errors
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        trigger[i] = fisher[i - 1] if i > period else fisher[i]
        
        normalized_prev = normalized
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (100 - 100 * sum(ATR) / highest_high - lowest_low)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            atr_sum = 0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
                atr_sum += tr
            chop[i] = 100 - 100 * atr_sum / (highest - lowest) / period
        else:
            chop[i] = 50
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, trigger = calculate_fisher(close, 9)
    chop = calculate_choppiness(high, low, close, 14)
    
    # KAMA fast/slow for crossover
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    kama_slow = calculate_kama(close, er_period=15, fast_period=3, slow_period=40)
    
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
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        is_trending = chop[i] < 38.2  # Trending regime
        is_ranging = chop[i] > 61.8   # Range/chop regime
        
        # KAMA trend direction
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # Fisher Transform signals (reversal detection)
        fisher_oversold = fisher[i] < -1.5 and trigger[i] > fisher[i]  # Bullish reversal
        fisher_overbought = fisher[i] > 1.5 and trigger[i] < fisher[i]  # Bearish reversal
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama_slow[i]
        price_below_kama = close[i] < kama_slow[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Fisher reversal + Daily bullish + KAMA bullish (trend + reversal)
        if fisher_oversold and daily_bullish and kama_bullish:
            new_signal = SIZE_ENTRY
        # Secondary: Fisher reversal + Daily bullish (regardless of KAMA)
        elif fisher_oversold and daily_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA crossover + Daily bullish + trending regime
        elif kama_bullish and daily_bullish and is_trending and price_above_kama:
            new_signal = SIZE_ENTRY
        # Quaternary: Price above KAMA + Daily bullish + ranging regime (mean reversion long)
        elif price_above_kama and daily_bullish and is_ranging and fisher[i] > -1.0:
            new_signal = SIZE_ENTRY
        # Fallback: Any bullish signal to ensure trade frequency
        elif daily_bullish and kama_bullish and fisher[i] > -2.0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Fisher reversal + Daily bearish + KAMA bearish (trend + reversal)
        if fisher_overbought and daily_bearish and kama_bearish:
            new_signal = -SIZE_ENTRY
        # Secondary: Fisher reversal + Daily bearish (regardless of KAMA)
        elif fisher_overbought and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA crossover + Daily bearish + trending regime
        elif kama_bearish and daily_bearish and is_trending and price_below_kama:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price below KAMA + Daily bearish + ranging regime (mean reversion short)
        elif price_below_kama and daily_bearish and is_ranging and fisher[i] < 1.0:
            new_signal = -SIZE_ENTRY
        # Fallback: Any bearish signal to ensure trade frequency
        elif daily_bearish and kama_bearish and fisher[i] < 2.0:
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