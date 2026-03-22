#!/usr/bin/env python3
"""
Experiment #574: 4h Dual HTF HMA + Choppiness Regime + RSI Entry + ATR Stop

Hypothesis: After 500+ failed experiments, the winning formula combines:
1. DUAL HTF bias (1d + 1w HMA) - stronger trend filter than single HTF
2. CHOPPINESS INDEX regime detection - switch between trend/mean-revert
3. RSI timing within regime - RSI<35 long in range, RSI>65 short in range
4. ASYMMETRIC logic - more aggressive shorts when 1w HMA bearish (2022 lesson)
5. ATR stoploss at 2.5x - protects against crypto crashes

Why 4h works:
- 4h has 6 bars/day = ~2190 bars/year = good trade frequency
- Captures multi-day trends without intraday noise
- Dual HTF (1d+1w) provides strong trend confirmation
- Choppiness filter avoids whipsaw in choppy markets (major failure mode)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_htf_hma_chop_regime_rsi_asymmetric_atr_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness formula
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.inf)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = chop.fillna(50).values
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG = 0.25   # Conservative long size
    SIZE_SHORT = 0.30  # More aggressive short (bear market bias)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # Both 1d and 1w must agree for strong signal
        bull_1d = close[i] > hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        strong_bull = bull_1d and bull_1w
        strong_bear = bear_1d and bear_1w
        neutral = not strong_bull and not strong_bear
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55  # Range market (slightly lower threshold for more trades)
        is_trending = chop_14[i] < 45  # Trend market
        
        # === RSI ENTRY SIGNALS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === ENTRY LOGIC (Regime-Adaptive + Asymmetric) ===
        new_signal = 0.0
        
        # LONG ENTRIES:
        # 1. In range market: RSI oversold + not strong bear
        if is_choppy and rsi_oversold and not strong_bear:
            new_signal = SIZE_LONG
        
        # 2. In trending market: RSI pullback + strong bull bias
        elif is_trending and rsi_14[i] < 50 and strong_bull:
            new_signal = SIZE_LONG
        
        # SHORT ENTRIES (more aggressive in bear markets):
        # 1. In range market: RSI overbought + not strong bull
        if is_choppy and rsi_overbought and not strong_bull:
            new_signal = -SIZE_SHORT
        
        # 2. In trending market: RSI rally + strong bear bias
        elif is_trending and rsi_14[i] > 50 and strong_bear:
            new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if dual HTF flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals