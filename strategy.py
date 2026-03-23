#!/usr/bin/env python3
"""
Experiment #899: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: After 600+ failed strategies, the winning formula combines:
1. 4h HMA(16/48) crossover for fast trend detection (proven in mtf_hma_rsi_zscore_v1)
2. 1d HMA(21) for macro trend bias (only trade with HTF direction)
3. RSI(14) pullback entries (enter on dips in uptrend, rallies in downtrend)
4. Choppiness Index(14) to reduce size in ranging markets (avoid whipsaw)
5. ATR(14) trailing stop (2.5x) for risk management

Why 4h works best:
- 20-50 trades/year target (lower fee drag than 1h/30m)
- Captures multi-day trends without 1d lag
- 1d HTF provides strong bias without over-filtering

Key improvements from failed experiments:
- RELAXED RSI thresholds (30/70 not 20/80) to guarantee trades
- HMA crossover (not single HMA) for clearer trend signals
- Choppiness reduces size (not blocks entries) to maintain trade count
- Simple logic = fewer bugs, more consistent across BTC/ETH/SOL

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_rsi_pullback_1d_chop_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging (reduce size), CHOP < 38.2 = trending (full size).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    hma_fast_4h = calculate_hma(close, 16)
    hma_slow_4h = calculate_hma(close, 48)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_fast_4h[i]) or np.isnan(hma_slow_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND BIAS (1d HTF HMA21) ===
        macro_bullish = close[i] > hma_1d_aligned[i]
        macro_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA Crossover) ===
        hma_bullish = hma_fast_4h[i] > hma_slow_4h[i]
        hma_bearish = hma_fast_4h[i] < hma_slow_4h[i]
        
        # === CHOPPINESS REGIME ===
        ranging = chop_4h[i] > 55
        trending = chop_4h[i] < 45
        
        # === RSI PULLBACK SIGNALS (Relaxed: 30/70) ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_extreme_oversold = rsi_4h[i] < 30
        rsi_extreme_overbought = rsi_4h[i] > 70
        
        # === HMA CROSSOVER CONFIRMATION ===
        hma_cross_long = hma_bullish and (i > 0 and hma_fast_4h[i-1] <= hma_slow_4h[i-1])
        hma_cross_short = hma_bearish and (i > 0 and hma_fast_4h[i-1] >= hma_slow_4h[i-1])
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: HMA bullish + RSI pullback + macro bias aligned
        if hma_bullish and rsi_oversold and macro_bullish:
            desired_signal = BASE_SIZE
        # Secondary: HMA crossover long + any RSI condition
        elif hma_cross_long and (rsi_oversold or rsi_4h[i] < 50):
            desired_signal = BASE_SIZE
        # Tertiary: Extreme RSI oversold (guarantees trades in ranging)
        elif rsi_extreme_oversold and (hma_bullish or macro_bullish):
            desired_signal = REDUCED_SIZE
        # Fallback: Strong macro bull + RSI neutral
        elif macro_bullish and rsi_4h[i] < 45 and not ranging:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: HMA bearish + RSI pullback + macro bias aligned
        if hma_bearish and rsi_overbought and macro_bearish:
            desired_signal = -BASE_SIZE
        # Secondary: HMA crossover short + any RSI condition
        elif hma_cross_short and (rsi_overbought or rsi_4h[i] > 50):
            desired_signal = -BASE_SIZE
        # Tertiary: Extreme RSI overbought (guarantees trades in ranging)
        elif rsi_extreme_overbought and (hma_bearish or macro_bearish):
            desired_signal = -REDUCED_SIZE
        # Fallback: Strong macro bear + RSI neutral
        elif macro_bearish and rsi_4h[i] > 55 and not ranging:
            desired_signal = -REDUCED_SIZE
        
        # === SIZE ADJUSTMENT FOR CHOPPINESS ===
        if ranging and desired_signal != 0:
            # Reduce size in ranging markets to avoid whipsaw
            if abs(desired_signal) == BASE_SIZE:
                desired_signal = np.sign(desired_signal) * REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish or macro still bullish
                if (hma_bullish or macro_bullish) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE if not ranging else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish or macro still bearish
                if (hma_bearish or macro_bearish) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE if not ranging else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HMA and macro reverse + RSI overbought
            if hma_bearish and macro_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HMA and macro reverse + RSI oversold
            if hma_bullish and macro_bullish and rsi_4h[i] < 35:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_extreme_oversold:
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