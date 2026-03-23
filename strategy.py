#!/usr/bin/env python3
"""
Experiment #646: 12h Primary + 1d HTF — Fisher + HMA + Choppiness (LOOSE ENTRIES)

Hypothesis: 12h timeframe with 1d HTF trend filter provides optimal balance between
signal quality and trade frequency. Previous 1d strategies failed with 0 trades due
to overly strict conditions. This version uses LOOSER Fisher thresholds and simpler
logic to ensure adequate trade generation.

Key innovations:
1. LOOSE Fisher thresholds (-1.0/+1.0 crosses) — ensures trades trigger
2. Simple 1d HMA trend bias — only 1 HTF filter, not multiple
3. Choppiness modulates position size, not entry eligibility
4. ATR trailing stop at 2.5x for risk management
5. Hold logic maintains positions through minor pullbacks

Why this should beat Sharpe=0.612 AND generate trades:
- 12h TF = fewer false signals than 4h/1h, more trades than 1d
- Single 1d HMA filter (not 1d+1w) = less condition conflict
- Fisher -1.0/+1.0 thresholds = triggers on moderate extremes, not just extreme
- Choppiness affects size not entry = still enter in all regimes
- Conservative sizing (0.25-0.30) survives crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_chop_loose_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    Long signal: Fisher crosses above -1.0 from below
    Short signal: Fisher crosses below +1.0 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_raw = (close[i] - ll) / range_val
        
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        price[i] = np.clip(price[i], -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use this to modulate position size, not block entries.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    fisher_12h, fisher_signal_12h = calculate_fisher_transform(high, low, close, period=9)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1d HMA for trend bias)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    
    # Base position sizes
    SIZE_BASE = 0.30
    SIZE_REDUCED = 0.20  # In choppy markets
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_12h[i]) or np.isnan(fisher_signal_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr_12h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS) ===
        # Long: Fisher crosses above -1.0 from below
        fisher_long_cross = (fisher_12h[i] > -1.0) and (fisher_signal_12h[i] <= -1.0)
        # Short: Fisher crosses below +1.0 from above
        fisher_short_cross = (fisher_12h[i] < 1.0) and (fisher_signal_12h[i] >= 1.0)
        
        # Fisher extreme levels (for additional entries)
        fisher_oversold = fisher_12h[i] < -1.5
        fisher_overbought = fisher_12h[i] > 1.5
        
        # === CHOPPINESS REGIME (modulates size, not entry) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # Determine position size based on regime
        current_size = SIZE_REDUCED if is_choppy else SIZE_BASE
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        # Long entries
        if fisher_long_cross:
            # Enter long on Fisher cross, HTF bias helps but not required
            if htf_bullish or not htf_bearish:
                desired_signal = current_size
        elif fisher_oversold:
            # Enter long on extreme oversold (mean reversion)
            desired_signal = current_size
        
        # Short entries
        elif fisher_short_cross:
            # Enter short on Fisher cross, HTF bias helps but not required
            if htf_bearish or not htf_bullish:
                desired_signal = -current_size
        elif fisher_overbought:
            # Enter short on extreme overbought (mean reversion)
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions still favorable ===
        # CRITICAL: This ensures we don't exit too quickly and generate trades
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if Fisher not extremely overbought
                if fisher_12h[i] < 1.8:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if Fisher not extremely oversold
                if fisher_12h[i] > -1.8:
                    desired_signal = -current_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = current_size
        elif desired_signal < 0:
            desired_signal = -current_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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