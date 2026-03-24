#!/usr/bin/env python3
"""
Experiment #658: 4h Primary + 1d HTF — KAMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: Instead of requiring Donchian breakouts (rare), use RSI pullback entries 
within the trend direction. This generates more consistent trades while maintaining 
quality. KAMA adapts to market efficiency, 1d HMA provides bias, Choppiness reduces 
size in ranges.

Key innovations:
1. 1d HMA(21) for primary trend bias - long above, short below
2. 4h KAMA(14) adaptive trend - slopes up/down for direction confirmation
3. RSI(14) 40-60 pullback zone - enters on dips in uptrend, rallies in downtrend
4. Choppiness Index(14) - reduces size by 50% when CHOP > 55 (choppy market)
5. ATR(14) 2.5x trailing stop - protects profits and limits losses
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Entry logic (LOOSE to ensure trades):
- LONG: close > 1d HMA AND KAMA sloping up (3 bars) AND RSI 35-55 (pullback)
- SHORT: close < 1d HMA AND KAMA sloping down (3 bars) AND RSI 45-65 (pullback)
- CHOP > 55: reduce size by 50% but don't block entries

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-30%
Timeframe: 4h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adjusts smoothing based on market efficiency"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        
        if volatility > 1e-10:
            er = price_change / volatility
        else:
            er = 0.0
        
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market chop vs trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low < 1e-10:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                    atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average for HTF"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=14, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION (3-bar slope) ===
        kama_bull = False
        kama_bear = False
        if i >= 3 and not np.isnan(kama[i-3]):
            kama_bull = kama[i] > kama[i-1] and kama[i-1] > kama[i-2] and kama[i-2] > kama[i-3]
            kama_bear = kama[i] < kama[i-1] and kama[i-1] < kama[i-2] and kama[i-2] < kama[i-3]
        
        # === RSI PULLBACK ZONE ===
        # Long: RSI 35-55 (pullback in uptrend, not oversold)
        # Short: RSI 45-65 (pullback in downtrend, not overbought)
        rsi_long_pullback = 35.0 <= rsi[i] <= 55.0
        rsi_short_pullback = 45.0 <= rsi[i] <= 65.0
        
        # === CHOPPINESS FILTER ===
        choppy_market = chop[i] > 55.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + KAMA up + RSI pullback
        if htf_bull and kama_bull and rsi_long_pullback:
            if choppy_market:
                desired_signal = SIZE_WEAK
            else:
                desired_signal = SIZE_BASE
        elif htf_bull and kama_bull and close[i] > kama[i]:
            # Weaker: just HTF + KAMA alignment without RSI pullback
            if not choppy_market:
                desired_signal = SIZE_WEAK
        
        # SHORT: HTF bear + KAMA down + RSI pullback
        elif htf_bear and kama_bear and rsi_short_pullback:
            if choppy_market:
                desired_signal = -SIZE_WEAK
            else:
                desired_signal = -SIZE_BASE
        elif htf_bear and kama_bear and close[i] < kama[i]:
            # Weaker: just HTF + KAMA alignment without RSI pullback
            if not choppy_market:
                desired_signal = -SIZE_WEAK
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals