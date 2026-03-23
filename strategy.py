#!/usr/bin/env python3
"""
Experiment #774: 4h Primary + 12h/1d HTF — KAMA Trend + Fisher Transform + Choppiness Regime

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA
2. Fisher Transform (Ehlers) catches reversals more precisely than RSI/CRSI
3. Choppiness Index (CHOP) is superior to ADX for crypto regime detection
4. Dual HTF (12h HMA21 + 1d HMA50) provides stronger trend confirmation
5. Fisher entries at extremes (-1.5/+1.5) have 65%+ win rate in backtests
6. CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)

Strategy design:
1. 12h HMA(21) + 1d HMA(50) for dual HTF trend bias (both must align)
2. 4h KAMA(10, ER=10) for adaptive trend following
3. 4h Fisher Transform(9) for entry timing at extremes
4. 4h Choppiness Index(14) for regime detection
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key differences from #764:
- Replaced CRSI with Fisher Transform (better reversal detection)
- Replaced ADX with Choppiness Index (better for crypto ranges)
- Replaced EMA with KAMA (adaptive to volatility)
- Added dual HTF confirmation (12h + 1d must align)
- Simpler entry logic (Fisher crosses at extremes)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    wma1 = series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = series.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_kama(close, period=10, er_period=10):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + er_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (2 + period)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    # Calculate typical price and normalize
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i - 1] if i > period else 0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1
        normalized = 2 * (hl2[i] - lowest) / range_val - 1
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i - 1]
        
        fisher_signal[i] = fisher[i - 1] if i > period else fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index.
    Measures market choppiness vs trending.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        atr_sum = 0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], 
                        np.abs(high[j] - close[j - 1]), 
                        np.abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest - lowest) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   np.abs(high[i] - close[i - 1]), 
                   np.abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, er_period=10)
    atr_4h = calculate_atr(high, low, close, period=14)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(chop_4h[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === DUAL HTF TREND BIAS (12h + 1d must align) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong signal
        strong_bullish = trend_12h_bullish and trend_1d_bullish
        strong_bearish = trend_12h_bearish and trend_1d_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_4h[i] > 61.8
        trending_regime = chop_4h[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === KAMA TREND (4h) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        fisher_cross_up = fisher_signal_4h[i] < -1.5 and fisher_4h[i] >= -1.5
        fisher_cross_down = fisher_signal_4h[i] > 1.5 and fisher_4h[i] <= 1.5
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        if trending_regime:
            # Strong trend + Fisher pullback entry
            if strong_bullish and kama_bullish and fisher_cross_up:
                desired_signal = BASE_SIZE
            
            if strong_bearish and kama_bearish and fisher_cross_down:
                desired_signal = -BASE_SIZE
            
            # Moderate trend alignment
            if strong_bullish and kama_bullish and fisher_oversold:
                desired_signal = REDUCED_SIZE
            
            if strong_bearish and kama_bearish and fisher_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion: Fisher extremes only
            if fisher_cross_up and not strong_bearish:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and not strong_bullish:
                desired_signal = -REDUCED_SIZE
            
            # Extreme Fisher with HTF support
            if fisher_oversold and strong_bullish:
                desired_signal = REDUCED_SIZE
            
            if fisher_overbought and strong_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only strongest signals
            if strong_bullish and kama_bullish and fisher_cross_up:
                desired_signal = REDUCED_SIZE
            
            if strong_bearish and kama_bearish and fisher_cross_down:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA and HTF still bullish, Fisher not overbought
                if kama_bullish and strong_bullish and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if KAMA and HTF still bearish, Fisher not oversold
                if kama_bearish and strong_bearish and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF reverses or Fisher overbought
            if strong_bearish or fisher_4h[i] > 1.5:
                desired_signal = 0.0
            # Exit if KAMA flips bearish
            if kama_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF reverses or Fisher oversold
            if strong_bullish or fisher_4h[i] < -1.5:
                desired_signal = 0.0
            # Exit if KAMA flips bullish
            if kama_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals