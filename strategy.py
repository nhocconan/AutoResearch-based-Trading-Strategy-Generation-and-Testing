#!/usr/bin/env python3
"""
Experiment #175: 6h Primary + 12h/1d HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. Volatility Spike Detection: ATR(7)/ATR(30) > 1.8 signals extreme vol expansion
2. Bollinger Band Extreme: Price at 2.5+ sigma indicates exhaustion
3. Regime Filter: Choppiness Index distinguishes range (mean revert) vs trend (breakout)
4. 12h/1d HTF Bias: Only trade mean reversion in direction of HTF trend

Why 6h: Middle ground between 4h (too many trades) and 12h (too few). 
Target 30-60 trades/year with 0.25-0.30 position sizing.

Key Edge: Vol spikes in crypto often mark temporary extremes that revert within 1-3 days.
Combined with HTF trend bias, this captures "panic buy/sell" reversals.

Position Sizing: 0.25 base, 0.30 for strong confluence (3+ signals align)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_bb_regime_12h1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with configurable std dev"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound (mean reversion favored)
    CHOP < 38.2 = trending (breakout favored)
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_vol_ratio(atr_short, atr_long):
    """Volatility expansion ratio: ATR(short) / ATR(long)"""
    n = len(atr_short)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    vol_ratio = calculate_vol_ratio(atr_7, atr_30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    hma_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(chop[i]):
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
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (both 12h and 1d agree)
        htf_strong_bull = htf_12h_bull and htf_1d_bull
        htf_strong_bear = htf_12h_bear and htf_1d_bear
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30)
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        bb_extreme_low = bb_position < 0.1  # Price in bottom 10% of BB
        bb_extreme_high = bb_position > 0.9  # Price in top 10% of BB
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range-bound (favor mean reversion)
        is_trending = chop[i] < 45.0  # Trending (favor breakout continuation)
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0
        
        # REGIME 1: CHOPPY + VOL SPIKE (Mean Reversion)
        if is_choppy and vol_spike:
            # Long: vol spike + BB extreme low + RSI oversold + HTF not strongly bear
            if bb_extreme_low and rsi_oversold and not htf_strong_bear:
                signal_strength = 2
                if htf_strong_bull or above_sma200:
                    signal_strength = 3
                if signal_strength >= 2:
                    desired_signal = SIZE_BASE if signal_strength == 2 else SIZE_STRONG
            
            # Short: vol spike + BB extreme high + RSI overbought + HTF not strongly bull
            elif bb_extreme_high and rsi_overbought and not htf_strong_bull:
                signal_strength = 2
                if htf_strong_bear or below_sma200:
                    signal_strength = 3
                if signal_strength >= 2:
                    desired_signal = -SIZE_BASE if signal_strength == 2 else -SIZE_STRONG
        
        # REGIME 2: TRENDING + VOL SPIKE (Breakout Continuation)
        elif is_trending and vol_spike:
            # Long: HTF bull + price above HMA21 + RSI not overbought
            if htf_strong_bull and close[i] > hma_21[i] and not rsi_overbought:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear + price below HMA21 + RSI not oversold
            elif htf_strong_bear and close[i] < hma_21[i] and not rsi_oversold:
                desired_signal = -SIZE_BASE
        
        # REGIME 3: TRANSITION (45-55 chop) - only strongest signals
        else:
            # Require vol spike + extreme BB + HTF alignment
            if vol_spike and bb_extreme_low and htf_strong_bull:
                desired_signal = SIZE_BASE
            elif vol_spike and bb_extreme_high and htf_strong_bear:
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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