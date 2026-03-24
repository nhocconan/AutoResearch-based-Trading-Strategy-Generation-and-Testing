#!/usr/bin/env python3
"""
Experiment #211: 6h Primary + 1w/1d HTF — Regime-Adaptive RSI/HMA Strategy

Hypothesis: 6h timeframe sits in the "sweet spot" between 4h (too noisy) and 12h (too slow).
Previous 6h experiments failed due to overly strict entry conditions (0 trades generated).
This version SIMPLIFIES entry logic to ensure trade generation while maintaining edge:

Regime Detection (CHOP 14):
- CHOP > 55 = choppy → mean reversion (RSI extremes 25/75)
- CHOP < 45 = trending → trend follow (HMA slope + price position)
- 45-55 = transition → reduced size, require HTF confirmation

HTF Filter (1d/1w):
- 1w HMA(21) = major trend bias (soft filter, not hard requirement)
- 1d HMA(50) = intermediate trend (used for confluence scoring)

Entry Logic (SIMPLIFIED to ensure trades):
- Choppy: RSI(7) < 25 + price > SMA100 → long (oversold in uptrend)
- Choppy: RSI(7) > 75 + price < SMA100 → short (overbought in downtrend)
- Trending: Price > HMA(21) + HMA slope up + 1d bias bull → long
- Trending: Price < HMA(21) + HMA slope down + 1d bias bear → short

Key changes from failed experiments:
- RSI period reduced from 14 to 7 (more signals)
- RSI thresholds widened from 15/85 to 25/75 (more triggers)
- HTF filter is soft confluence, not hard requirement
- SMA100 instead of SMA200 (more responsive)
- Start loop at index 150 instead of 250 (earlier trading)

Position sizing: 0.25 base, 0.30 strong (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 40-80 trades/year, Sharpe > 0.40, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_rsi_hma_1w1d_v2"
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

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope over lookback period"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback] * 100.0
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_slope = calculate_hma_slope(hma_6h, lookback=5)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for more signals
    rsi_14 = calculate_rsi(close, period=14)
    sma_100 = calculate_sma(close, 100)
    sma_50 = calculate_sma(close, 50)
    
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
    
    for i in range(150, n):  # Start earlier to ensure trades
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (soft filters - confluence scoring) ===
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Count HTF confluence (0-2)
        htf_bull_score = int(htf_1w_bull) + int(htf_1d_bull)
        htf_bear_score = int(htf_1w_bear) + int(htf_1d_bear)
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        hma_slope_up = not np.isnan(hma_slope[i]) and hma_slope[i] > 0.5
        hma_slope_down = not np.isnan(hma_slope[i]) and hma_slope[i] < -0.5
        
        # === SMA FILTERS ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0
        
        # REGIME 1: CHOPPY (mean reversion)
        if is_choppy:
            # Long: RSI oversold + above SMA100 (uptrend pullback)
            if rsi_oversold and above_sma100:
                desired_signal = SIZE_BASE
                signal_strength = 1
                if htf_bull_score >= 1:
                    desired_signal = SIZE_STRONG
                    signal_strength = 2
            
            # Short: RSI overbought + below SMA100 (downtrend rally)
            elif rsi_overbought and below_sma100:
                desired_signal = -SIZE_BASE
                signal_strength = 1
                if htf_bear_score >= 1:
                    desired_signal = -SIZE_STRONG
                    signal_strength = 2
        
        # REGIME 2: TRENDING (trend following)
        elif is_trending:
            # Long: HMA bull + slope up + HTF bias
            if hma_bull and hma_slope_up:
                if htf_bull_score >= 1:
                    desired_signal = SIZE_STRONG
                    signal_strength = 2
                elif hma_bull and above_sma100:
                    desired_signal = SIZE_BASE
                    signal_strength = 1
            
            # Short: HMA bear + slope down + HTF bias
            elif hma_bear and hma_slope_down:
                if htf_bear_score >= 1:
                    desired_signal = -SIZE_STRONG
                    signal_strength = 2
                elif hma_bear and below_sma100:
                    desired_signal = -SIZE_BASE
                    signal_strength = 1
        
        # REGIME 3: TRANSITION (45-55 chop) - require stronger confirmation
        else:
            # Only enter with HTF confirmation
            if hma_bull and htf_bull_score >= 1 and above_sma100:
                desired_signal = SIZE_BASE
                signal_strength = 1
            elif hma_bear and htf_bear_score >= 1 and below_sma100:
                desired_signal = -SIZE_BASE
                signal_strength = 1
        
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