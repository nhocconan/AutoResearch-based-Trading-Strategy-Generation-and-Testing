#!/usr/bin/env python3
"""
Experiment #1106: 12h Primary + 1d HTF — Dual Regime (Choppiness + HMA) with RSI Entries

Hypothesis: After exp#1096 failed (Sharpe=-0.321), the issue was single-regime logic.
Key insight from research: Choppiness Index regime filter gave ETH Sharpe +0.923.

Strategy:
1. CHOP(14) > 61.8 = RANGE regime → mean reversion (RSI extremes)
2. CHOP(14) < 38.2 = TREND regime → trend follow (1d HMA direction)
3. CHOP between = neutral → reduce size or stay flat
4. 1d HMA provides macro bias filter
5. Loose RSI thresholds (35/65) ensure adequate trade frequency
6. ATR 2.5x trailing stoploss

Why this should work:
- Dual regime adapts to market conditions (range vs trend)
- 12h naturally limits trades to 20-50/year
- Research shows CHOP regime filter works on ETH (Sharpe +0.923)
- Simpler than triple-regime but more adaptive than single-regime

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 25-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_rsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = Range/Chop (mean reversion favored)
    CHOP < 38.2 = Trend (trend following favored)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    price_range = highest - lowest
    mask = price_range > 1e-10
    
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    chop[~mask] = 50.0  # Default to neutral when no range
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop[i]
        is_range = chop_value > 55.0  # Range/choppy market
        is_trend = chop_value < 45.0  # Trending market
        # Between 45-55 = neutral/transition
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI SIGNALS ===
        rsi_value = rsi_12h[i]
        rsi_oversold = rsi_value < 40.0
        rsi_overbought = rsi_value > 60.0
        rsi_extreme_low = rsi_value < 30.0
        rsi_extreme_high = rsi_value > 70.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === RANGE REGIME: Mean Reversion ===
        if is_range:
            # Long when RSI oversold in range
            if rsi_oversold and macro_bull:
                desired_signal = current_size
            # Short when RSI overbought in range
            elif rsi_overbought and macro_bear:
                desired_signal = -current_size
            # Extreme RSI overrides macro
            elif rsi_extreme_low:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_high:
                desired_signal = -REDUCED_SIZE
        
        # === TREND REGIME: Trend Following ===
        elif is_trend:
            # Long pullback in uptrend
            if macro_bull and rsi_oversold:
                desired_signal = current_size
            # Short pullback in downtrend
            elif macro_bear and rsi_overbought:
                desired_signal = -current_size
            # Strong trend continuation
            elif macro_bull and rsi_value > 50.0 and rsi_value < 65.0:
                desired_signal = REDUCED_SIZE
            elif macro_bear and rsi_value < 50.0 and rsi_value > 35.0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME: Reduced size or flat ===
        else:
            # Only take extreme signals in neutral
            if rsi_extreme_low and macro_bull:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_high and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull or range with RSI not overbought
                if (macro_bull and not rsi_overbought) or (is_range and rsi_value < 65.0):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear or range with RSI not oversold
                if (macro_bear and not rsi_oversold) or (is_range and rsi_value > 35.0):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear or RSI very overbought
            if macro_bear and is_trend:
                desired_signal = 0.0
            elif rsi_extreme_high:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull or RSI very oversold
            if macro_bull and is_trend:
                desired_signal = 0.0
            elif rsi_extreme_low:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals