#!/usr/bin/env python3
"""
Experiment #315: 6h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime v1

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Combined with Choppiness Index regime detection and 12h/1d HMA trend bias for 6h timeframe.

Why 6h is different:
- Middle ground between 4h (too noisy) and 12h (too slow)
- Captures multi-day swings without excessive trade frequency
- Target: 30-60 trades/year (6-12 per symbol per year)

Key innovations:
1. FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
   Catches reversals better than RSI in bear markets (proven in literature)
2. CHOPPINESS REGIME: CHOP>60 = mean revert (Fisher extremes), CHOP<45 = trend follow (HMA break)
3. 12h HMA TREND BIAS: Only take longs when 12h HMA bullish, shorts when bearish
4. 1d HMA CONFIRMATION: Boost size when 1d aligned with 12h
5. ASYMMETRIC SIZING: 0.25 base, 0.30 when HTF aligned (discrete levels)
6. ATR STOPLOSS: 2.5x ATR from entry, signal→0 when hit

Regime Logic:
- Choppy (CHOP>60): Fisher mean reversion at extremes (-1.5/+1.5)
- Trending (CHOP<45): HMA breakout + HTF confirmation
- Transition (45-60): Use previous regime memory (hysteresis)

Position sizing: 0.25 base, 0.30 when 12h+1d aligned
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_hma_12h1d_v1"
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
    CHOP > 61.8 = choppy/range bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI
    
    Formula:
    1. Calculate typical price: (H+L)/2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Fisher: 0.5 * ln((1+x)/(1-x))
    4. Signal line: 1-period lag of Fisher
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal = np.zeros(n)
    signal[:] = np.nan
    
    for i in range(period, n):
        # Find highest and lowest over lookback
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize to -1 to +1
        x = ((typical[i] - lowest) / price_range) * 2.0 - 1.0
        
        # Clamp to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag)
        if i > 0 and not np.isnan(fisher[i-1]):
            signal[i] = fisher[i-1]
    
    return fisher, signal

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 45.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS (12h + 1d) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HTF alignment (both same direction)
        htf_aligned_bull = htf_12h_bull and htf_1d_bull
        htf_aligned_bear = htf_12h_bear and htf_1d_bear
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = False
        fisher_short_signal = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher_signal[i]):
            # Fisher crossing above -1.5 from below
            if fisher_signal[i] < -1.5 and fisher[i] >= -1.5:
                fisher_long_signal = True
            # Fisher crossing below +1.5 from above
            if fisher_signal[i] > 1.5 and fisher[i] <= 1.5:
                fisher_short_signal = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Fisher mean reversion)
        if current_regime == 2:
            # Long: Fisher oversold reversal + above SMA200 + 12h bull bias
            if fisher_long_signal and above_sma200 and htf_12h_bull:
                desired_signal = SIZE_STRONG if htf_aligned_bull else SIZE_BASE
            
            # Short: Fisher overbought reversal + below SMA200 + 12h bear bias
            elif fisher_short_signal and below_sma200 and htf_12h_bear:
                desired_signal = -SIZE_STRONG if htf_aligned_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (HMA breakout with HTF confirmation)
        elif current_regime == 1:
            # Long: 6h HMA bull + 12h HMA bull + 1d HMA bull
            if hma_bull and htf_12h_bull and htf_1d_bull:
                # Only enter on Fisher confirmation (pullback entry)
                if fisher[i] < 0.5:  # Not overbought
                    desired_signal = SIZE_STRONG if htf_aligned_bull else SIZE_BASE
            
            # Short: 6h HMA bear + 12h HMA bear + 1d HMA bear
            elif hma_bear and htf_12h_bear and htf_1d_bear:
                # Only enter on Fisher confirmation (pullback entry)
                if fisher[i] > -0.5:  # Not oversold
                    desired_signal = -SIZE_STRONG if htf_aligned_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals