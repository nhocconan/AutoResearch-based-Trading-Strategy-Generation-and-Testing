#!/usr/bin/env python3
"""
Experiment #342: 12h Primary + 1d/1w HTF — Dual Regime with Asymmetric Entries

Hypothesis: Previous 12h failures (#336) had too many regime filters killing trade count.
This strategy simplifies entry logic while maintaining regime awareness:

1. 1w HMA(21) as ULTRA-MACRO BIAS (only trade in direction of weekly trend)
2. 1d HMA(21) as MACRO BIAS (secondary filter for entry direction)
3. 12h Choppiness Index for regime (CHOP>55=mean revert, CHOP<45=trend follow)
4. 12h HMA(16/48) crossover for trend entries
5. 12h RSI(14) for mean reversion entries at extremes
6. ATR(14) trailing stop at 2.5x

KEY INSIGHT: Asymmetric entries based on 1w/1d trend alignment.
- When 1w AND 1d both bullish: only longs, easier entry (RSI 35-65)
- When 1w AND 1d both bearish: only shorts, easier entry (RSI 35-65)
- When mixed: smaller size, stricter entry (RSI 25-35 or 65-75)

TARGET: 25-45 trades/year on 12h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_1d1w_bias_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    diff = 2 * wma_half - wma_full
    hma = diff.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP (>61.8) = choppy/ranging, Low CHOP (<38.2) = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # HMA for trend detection on 12h
    hma_fast_12h = calculate_hma(close, period=16)
    hma_slow_12h = calculate_hma(close, period=48)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (conservative)
    REDUCED_SIZE = 0.18  # Reduced size when HTF signals mixed
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_fast_12h[i]) or np.isnan(hma_slow_12h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ULTRA-MACRO BIAS (1w HMA - HARDEST FILTER) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === HTF ALIGNMENT CHECK ===
        # Strong bull: both 1w and 1d bullish
        # Strong bear: both 1w and 1d bearish
        # Mixed: conflicting signals (use reduced size)
        strong_bull = price_above_hma_1w and price_above_hma_1d
        strong_bear = price_below_hma_1w and price_below_hma_1d
        mixed_signal = not strong_bull and not strong_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = mean reversion
        is_trending = chop[i] < 45.0  # Low choppiness = trend following
        
        # === TREND SIGNAL (12h HMA crossover) ===
        hma_bullish = hma_fast_12h[i] > hma_slow_12h[i]
        hma_bearish = hma_fast_12h[i] < hma_slow_12h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        current_size = BASE_SIZE if (strong_bull or strong_bear) else REDUCED_SIZE
        
        if is_trending:
            # TREND REGIME: HMA crossover entries
            
            # LONG: 1w bullish + (1d bullish OR mixed) + HMA bullish
            if price_above_hma_1w and hma_bullish:
                # Entry on pullback (RSI 40-60) or continuation (RSI 50-65)
                if 40 <= rsi_14[i] <= 65:
                    desired_signal = current_size
            
            # SHORT: 1w bearish + (1d bearish OR mixed) + HMA bearish
            elif price_below_hma_1w and hma_bearish:
                # Entry on pullback (RSI 35-60) or continuation (RSI 35-50)
                if 35 <= rsi_14[i] <= 60:
                    desired_signal = -current_size
        
        elif is_choppy:
            # CHOPPY REGIME: Mean reversion at extremes
            
            # LONG: 1w bullish + RSI oversold
            if price_above_hma_1w and rsi_14[i] < 35:
                desired_signal = current_size * 0.8
            
            # SHORT: 1w bearish + RSI overbought
            elif price_below_hma_1w and rsi_14[i] > 65:
                desired_signal = -current_size * 0.8
        
        else:
            # NEUTRAL REGIME: Wait for clearer signals
            # Only enter on strong RSI extremes with HTF alignment
            
            if strong_bull and rsi_14[i] < 30:
                desired_signal = REDUCED_SIZE
            
            elif strong_bear and rsi_14[i] > 70:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === RSI EXTREME EXIT (overbought/oversold reversal) ===
        if in_position and position_side > 0 and rsi_14[i] > 75:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend still valid for longs
            if position_side > 0 and hma_bullish and price_above_hma_1w:
                desired_signal = current_size
            # Check if trend still valid for shorts
            elif position_side < 0 and hma_bearish and price_below_hma_1w:
                desired_signal = -current_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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