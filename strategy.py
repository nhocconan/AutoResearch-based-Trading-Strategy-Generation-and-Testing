#!/usr/bin/env python3
"""
Experiment #1631: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Regime Filter

Hypothesis: 4h timeframe with simplified entry conditions will generate sufficient trades
while maintaining quality. Previous 12h strategies failed due to too few trades or negative Sharpe.

Key learnings from failures:
- #1626 (12h) had negative Sharpe (-0.095) - whipsaw in choppy markets
- Multiple strategies got 0 trades - entry conditions too strict
- 4h works better than 12h for trade frequency (see #1619, #1621 kept)

This strategy:
1. 4h primary with 1d HMA for trend bias + 1w for major trend filter
2. HMA(10/25) crossover for entry timing (proven in best lineage)
3. RSI(14) pullback entries - LOOSE thresholds (30-70) to ensure trades
4. CHOP regime filter - soft filter only, not hard gate
5. ATR(14) trailing stop at 2.5x
6. Size = 0.30 (discrete levels)

Why this should work:
- 4h targets 20-50 trades/year = optimal fee/trade balance
- Loose RSI thresholds ensure we get trades (avoid 0-trade failure)
- Dual HTF (1d+1w) provides strong trend filter without over-complicating
- CHOP as soft filter reduces whipsaw but doesn't block all entries

Timeframe: 4h (required for experiment #1631)
HTF: 1d HMA + 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_dual_htf_chop_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # HMA for trend following (fast and slow)
    hma_fast = calculate_hma(close, period=10)
    hma_slow = calculate_hma(close, period=25)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
        # Use as soft filter - reduce size in choppy, full size in trending
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === TREND BIAS (1d + 1w HMA) ===
        # Weekly provides major trend, daily provides intermediate trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        strong_bull = weekly_bull and daily_bull
        strong_bear = weekly_bear and daily_bear
        # Weak/neutral when they disagree
        neutral_bias = (weekly_bull and daily_bear) or (weekly_bear and daily_bull)
        
        # === HMA CROSSOVER ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # Check previous bar for crossover detection
        hma_bull_prev = False
        hma_bear_prev = False
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            hma_bull_prev = hma_fast[i-1] > hma_slow[i-1]
            hma_bear_prev = hma_fast[i-1] < hma_slow[i-1]
        
        # Bullish crossover (fast crosses above slow)
        hma_cross_up = hma_bull and not hma_bull_prev
        # Bearish crossover (fast crosses below slow)
        hma_cross_down = hma_bear and not hma_bear_prev
        
        # === RSI PULLBACK (LOOSE thresholds to ensure trades) ===
        # Long: RSI not overbought (below 70), ideally pulling back (30-60)
        rsi_ok_long = rsi[i] < 70.0
        rsi_pullback_long = 30.0 < rsi[i] < 60.0
        # Short: RSI not oversold (above 30), ideally bouncing (40-70)
        rsi_ok_short = rsi[i] > 30.0
        rsi_pullback_short = 40.0 < rsi[i] < 70.0
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        signal_strength = 1.0  # Can reduce in choppy markets
        
        # Reduce size in choppy regime
        if is_choppy:
            signal_strength = 0.6  # 60% size in choppy
        
        # TREND FOLLOWING MODE (strong bias + trending regime)
        if strong_bull and is_trending:
            # Long: HMA bullish + RSI ok
            if hma_bull and rsi_ok_long:
                desired_signal = BASE_SIZE * signal_strength
        elif strong_bear and is_trending:
            # Short: HMA bearish + RSI ok
            if hma_bear and rsi_ok_short:
                desired_signal = -BASE_SIZE * signal_strength
        
        # PULLBACK MODE (neutral bias or choppy - mean reversion)
        elif neutral_bias or is_choppy:
            # Long on pullback with HMA support
            if hma_bull and rsi_pullback_long:
                desired_signal = BASE_SIZE * signal_strength
            # Short on bounce with HMA resistance
            elif hma_bear and rsi_pullback_short:
                desired_signal = -BASE_SIZE * signal_strength
        
        # CROSSOVER MODE (any regime - catch momentum shifts)
        if hma_cross_up and rsi_ok_long and not strong_bear:
            desired_signal = max(desired_signal, BASE_SIZE * signal_strength)
        elif hma_cross_down and rsi_ok_short and not strong_bull:
            desired_signal = min(desired_signal, -BASE_SIZE * signal_strength)
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
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