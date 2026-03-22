#!/usr/bin/env python3
"""
Experiment #278: 30m KAMA Trend with 4h HMA Bias - Simplified for Trade Frequency

Hypothesis: After 277 failed experiments, the clearest pattern is:
- Complex entry conditions = 0 trades (#266, #277 both had Sharpe=0.000)
- 30m timeframe needs LOOSE entries to generate >=10 trades/symbol
- KAMA adapts to volatility better than EMA (reduces whipsaw)
- 4h HMA provides strong directional bias without over-filtering

Key differences from FAILED 30m strategies:
- #266 (Supertrend+4h HMA+ADX): 0 trades - ADX filter too restrictive
- #272 (KAMA pullback+4h HMA): -46.9% - pullback logic was wrong
- #277 (Supertrend+4h HMA+ADX+RSI): 0 trades - way too many filters

This strategy SIMPLIFIES to ensure trade frequency:
1. 30m KAMA(14) - adaptive trend, less lag than EMA
2. 4h HMA(21) - directional bias via mtf_data (call ONCE before loop)
3. NO ADX filter (was killing trade frequency)
4. NO RSI filter (consistently failed in #251, #254, #259)
5. Entry: 4h bias + price crosses KAMA (simple crossover)
6. 2.5*ATR trailing stoploss
7. Position size: 0.25-0.35 discrete levels

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data(prices, '4h') - ONCE before loop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_simple_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_REDUCED = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA CROSSOVER SIGNAL ===
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        prev_price_above = close[i-1] > kama[i-1] if i > 0 else False
        prev_price_below = close[i-1] < kama[i-1] if i > 0 else False
        
        crossover_long = price_above_kama and not prev_price_above
        crossover_short = price_below_kama and not prev_price_below
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        position_size = SIZE_REDUCED if high_volatility else SIZE_BASE
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        stoploss_triggered = False
        
        # Check stoploss on EXISTING position FIRST (takes precedence)
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
                    stoploss_triggered = True
        
        # Only check entry if stoploss didn't trigger
        if not stoploss_triggered:
            # LONG: 4h bias up + KAMA crossover up
            if bull_trend_4h and crossover_long:
                new_signal = position_size
            
            # SHORT: 4h bias down + KAMA crossover down
            if bear_trend_4h and crossover_short:
                new_signal = -position_size
            
            # Exit if HTF bias reverses against position
            if in_position and new_signal != 0.0:
                if position_side > 0 and bear_trend_4h:
                    new_signal = 0.0
                if position_side < 0 and bull_trend_4h:
                    new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals