#!/usr/bin/env python3
"""
Experiment #183: 6h Primary + 1d/1w HTF — Regime Adaptive RSI + HMA Trend

Hypothesis: 6h timeframe sits between 4h and 12h - can capture multi-day swings 
without the noise of lower TFs. Previous 6h attempts failed due to:
- Too many filters (0 trades)
- Volume-based entries (unreliable on 6h)
- Weekly pivot complexity (failed)

This version simplifies:
1. Regime detection via Choppiness Index (proven in #172 12h)
2. RSI extremes for mean reversion in chop (simpler than CRSI)
3. HMA crossovers for trend following
4. 1d + 1w HTF for major trend bias
5. LOOSE entry thresholds to ensure trades generate

Entry Logic:
- CHOPPY (CHOP>55): RSI<35 long, RSI>65 short (mean reversion)
- TRENDING (CHOP<45): HMA crossover + HTF alignment (trend follow)
- Size: 0.25 base, 0.30 strong conviction
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.167, trades≥30 train, trades≥3 test per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_rsi_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=13)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_fast = calculate_hma(close, period=13)
    hma_slow = calculate_hma(close, period=34)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    rsi_fast = calculate_rsi(close, period=7)
    sma_100 = calculate_sma(close, 100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(rsi_fast[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        # 1d trend
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w trend (major bias)
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Combined HTF bias strength
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_neutral = not htf_strong_bull and not htf_strong_bear
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === PRIMARY TREND ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === RSI VALUES ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_low = rsi_fast[i] < 30.0
        rsi_extreme_high = rsi_fast[i] > 70.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with RSI)
        if is_choppy:
            # Long: RSI oversold + above SMA100 + HTF not strongly bear
            if rsi_oversold and above_sma100 and not htf_strong_bear:
                desired_signal = SIZE_BASE
            
            # Short: RSI overbought + below SMA100 + HTF not strongly bull
            elif rsi_overbought and below_sma100 and not htf_strong_bull:
                desired_signal = -SIZE_BASE
            
            # Extreme RSI entries (stronger signal)
            elif rsi_extreme_low and above_sma100:
                desired_signal = SIZE_STRONG
            elif rsi_extreme_high and below_sma100:
                desired_signal = -SIZE_STRONG
        
        # REGIME 2: TRENDING (HMA crossover + HTF alignment)
        elif is_trending:
            # Long: HMA bull crossover + HTF bull alignment
            if hma_bull and htf_strong_bull:
                desired_signal = SIZE_STRONG
            elif hma_bull and htf_1d_bull:
                desired_signal = SIZE_BASE
            
            # Short: HMA bear crossover + HTF bear alignment
            elif hma_bear and htf_strong_bear:
                desired_signal = -SIZE_STRONG
            elif hma_bear and htf_1d_bear:
                desired_signal = -SIZE_BASE
        
        # REGIME 3: TRANSITION (45-55 chop) - use HMA direction with HTF confirm
        else:
            if hma_bull and htf_1d_bull and above_sma100:
                desired_signal = SIZE_BASE * 0.7
            elif hma_bear and htf_1d_bear and below_sma100:
                desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.7
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals