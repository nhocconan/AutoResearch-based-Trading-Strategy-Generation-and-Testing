#!/usr/bin/env python3
"""
Experiment #076: 12h Primary + 1d HTF — HMA Trend + RSI Pullback with Choppiness Filter

Hypothesis: 12h timeframe with 1d macro bias using HMA crossover for trend direction,
RSI pullback entries in trending markets, and Choppiness Index to reduce whipsaw in
ranging conditions. Simpler logic than previous attempts to ensure trade generation.

Key innovations:
1) HMA(16/48) crossover on 12h for faster trend detection than single HMA
2) 1d HMA(21) for macro bias - only trade in direction of daily trend
3) RSI(14) pullback: enter when RSI dips to 40-50 in uptrend, 50-60 in downtrend
4) Choppiness Index filter: reduce size (not block) when CHOP > 50
5) ATR(14) trailing stoploss at 2.5x
6) Relaxed entry thresholds to ensure 20-50 trades/year target

Why this should work:
- 12h proven timeframe (exp #066 failed but had too strict entries)
- HMA crossover reacts faster than single HMA for trend changes
- RSI pullback entries catch continuations, not reversals
- 1d filter prevents counter-trend trades in bear markets (2025 test period)
- Simpler logic = more trades = better statistical significance

Position size: 0.30 trend, 0.20 ranging (discrete)
Stoploss: 2.5*ATR trailing
Target: 20-50 trades/year, Sharpe > 0.486
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_rsi_pullback_1d_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # HMA crossover on 12h
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_RANGE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0
        is_trending = chop_value < 45.0
        
        # === RSI PULLBACK SIGNALS ===
        # In uptrend: buy pullback when RSI dips to 40-55
        rsi_pullback_long = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 55.0)
        # In downtrend: sell pullback when RSI rallies to 45-60
        rsi_pullback_short = (rsi_14[i] >= 45.0) and (rsi_14[i] <= 60.0)
        
        # === HMA CROSSOVER CONFIRMATION ===
        hma_cross_long = hma_bullish and (hma_16[i-1] <= hma_48[i-1] if i > 0 else False)
        hma_cross_short = hma_bearish and (hma_16[i-1] >= hma_48[i-1] if i > 0 else False)
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # LONG entries
        if hma_bullish and price_above_hma_1d:
            if is_trending:
                # Trending: enter on RSI pullback
                if rsi_pullback_long:
                    new_signal = POSITION_SIZE_TREND
                # Or on fresh HMA crossover
                elif hma_cross_long:
                    new_signal = POSITION_SIZE_TREND
            elif is_ranging:
                # Ranging: smaller size on RSI extremes
                if rsi_14[i] < 35.0:
                    new_signal = POSITION_SIZE_RANGE
        
        # SHORT entries
        elif hma_bearish and price_below_hma_1d:
            if is_trending:
                # Trending: enter on RSI pullback
                if rsi_pullback_short:
                    new_signal = -POSITION_SIZE_TREND
                # Or on fresh HMA crossover
                elif hma_cross_short:
                    new_signal = -POSITION_SIZE_TREND
            elif is_ranging:
                # Ranging: smaller size on RSI extremes
                if rsi_14[i] > 65.0:
                    new_signal = -POSITION_SIZE_RANGE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold long if RSI not overbought and trend intact
            if position_side > 0 and rsi_14[i] < 70.0 and hma_bullish:
                new_signal = signals[i-1] if i > 0 else 0.0
            # Hold short if RSI not oversold and trend intact
            elif position_side < 0 and rsi_14[i] > 30.0 and hma_bearish:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if hma_bearish or price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_bullish or price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals