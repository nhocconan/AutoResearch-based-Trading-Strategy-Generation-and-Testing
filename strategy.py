#!/usr/bin/env python3
"""
Experiment #102: 12h Primary + 1d HTF — Simplified RSI Mean Reversion with Trend Filter

Hypothesis: Previous 12h strategies failed due to overly complex position tracking and 
conflicting entry conditions. This version uses SIMPLER signal generation with proven 
12h/1d MTF structure but cleaner logic.

Key changes from failures:
1) REMOVE complex position tracking state machine - signals flow naturally
2) SIMPLER entry: RSI extremes (RSI<40 long, RSI>60 short) with 1d HMA trend filter
3) Choppiness is OPTIONAL boost only - doesn't block trades
4) ATR stoploss from ENTRY price (not trailing) - cleaner exit logic
5) Discrete sizing: 0.0, ±0.25, ±0.30 only
6) LOOSEN thresholds to ensure 25-50 trades/year

Why this should work:
- Simpler signal logic = fewer bugs, more trades
- 1d HMA prevents counter-trend trades in 2025 bear market
- RSI mean reversion works in both bull and bear regimes
- 12h timeframe naturally limits trades (no fee drag)
- Proven MTF structure from successful strategies

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 2.5*ATR from entry price
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_meanrev_1d_hma_chop_v1"
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
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track entry prices for stoploss
    long_entry_price = np.zeros(n)
    short_entry_price = np.zeros(n)
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME (OPTIONAL BOOST) ===
        chop_ranging = chop_14[i] > 55.0  # ranging market favors mean reversion
        chop_trending = chop_14[i] < 45.0  # trending market
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === RSI ENTRY SIGNALS (LOOSE for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0  # long entry
        rsi_overbought = rsi_14[i] > 60.0  # short entry
        rsi_extreme_long = rsi_14[i] < 30.0  # strong long
        rsi_extreme_short = rsi_14[i] > 70.0  # strong short
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d uptrend + RSI oversold ---
        # Primary: 1d HMA bullish + RSI < 40
        if price_above_hma_1d and rsi_oversold:
            new_signal = POSITION_SIZE_BASE
            # Boost if ranging regime (mean reversion works better)
            if chop_ranging:
                new_signal = POSITION_SIZE_MAX
            # Boost if extreme RSI
            if rsi_extreme_long:
                new_signal = POSITION_SIZE_MAX
            # Boost if above SMA200 (strong uptrend pullback)
            if price_above_sma200:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 1d downtrend + RSI overbought ---
        # Primary: 1d HMA bearish + RSI > 60
        if price_below_hma_1d and rsi_overbought:
            new_signal = -POSITION_SIZE_BASE
            # Boost if ranging regime (mean reversion works better)
            if chop_ranging:
                new_signal = -POSITION_SIZE_MAX
            # Boost if extreme RSI
            if rsi_extreme_short:
                new_signal = -POSITION_SIZE_MAX
            # Boost if below SMA200 (strong downtrend rally)
            if price_below_sma200:
                new_signal = -POSITION_SIZE_MAX
        
        # === STOPLOSS CHECK (2.5 * ATR from entry) ===
        # Check if we have an active position from previous bar
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if prev_signal > 0:  # Long position active
            # Track entry price (first bar of position)
            if prev_signal > 0 and (i == 250 or signals[i-2] <= 0 if i > 250 else True):
                long_entry_price[i] = close[i]
            else:
                long_entry_price[i] = long_entry_price[i-1] if i > 0 else close[i]
            
            stop_price = long_entry_price[i] - 2.5 * atr_14[i]
            if close[i] < stop_price:
                new_signal = 0.0  # Stoploss hit
        
        if prev_signal < 0:  # Short position active
            # Track entry price (first bar of position)
            if prev_signal < 0 and (i == 250 or signals[i-2] >= 0 if i > 250 else True):
                short_entry_price[i] = close[i]
            else:
                short_entry_price[i] = short_entry_price[i-1] if i > 0 else close[i]
            
            stop_price = short_entry_price[i] + 2.5 * atr_14[i]
            if close[i] > stop_price:
                new_signal = 0.0  # Stoploss hit
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if prev_signal > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0  # Long take profit
        
        if prev_signal < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0  # Short take profit
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish
        if prev_signal > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if prev_signal < 0 and price_above_hma_1d:
            new_signal = 0.0
        
        signals[i] = new_signal
    
    return signals