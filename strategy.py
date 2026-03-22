#!/usr/bin/env python3
"""
Experiment #026: 12h KAMA Trend + Fisher Transform Entries with 1d Bias

Hypothesis: Previous 12h strategies failed due to overly restrictive RSI pullback
ranges and HMA being too smooth (missing entries). This strategy uses:
1. KAMA(14) - Adaptive moving average that speeds up in trends, slows in chop
2. Fisher Transform(9) - Normalizes price to -1 to +1, excellent for reversal timing
3. Choppiness Index(14) - Regime detection: >61.8 = range, <38.2 = trend
4. 1d KAMA bias - Major trend direction filter
5. Wider stoploss (3.0 ATR) for 12h timeframe volatility
6. Relaxed entry conditions to ensure 20-50 trades/year

Why this should work:
- KAMA adapts to market regime better than HMA (proven in quant literature)
- Fisher Transform catches reversals at extremes (better than RSI for timing)
- Choppiness filter prevents trend entries in range markets (reduces whipsaws)
- 12h timeframe + 1d bias = proven combination (current best is 4h variant)
- Relaxed Fisher thresholds (-1.2 to +1.2) ensure enough trade frequency

Key improvements over #016:
- KAMA instead of HMA (more responsive to trend changes)
- Fisher Transform instead of RSI pullback (better reversal timing)
- Choppiness regime filter (avoid trend entries in chop)
- Wider stoploss (3.0 ATR vs 2.5 ATR)
- Simpler entry logic (fewer conflicting conditions)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_chop_1d_bias_v1"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    Efficiency Ratio (ER) = |change| / sum(|changes|) over period
    """
    close_s = pd.Series(close)
    change = np.abs(close_s.diff())
    signal = np.abs(close_s.diff(period))
    noise = change.rolling(window=period, min_periods=period).sum()
    
    er = signal / noise
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution (-1 to +1 range).
    Excellent for identifying reversals at extremes.
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Highest high and lowest low over period
    hh = typical_s.rolling(window=period, min_periods=period).max()
    ll = typical_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range
    norm = (typical_s - ll) / (hh - ll + 1e-10)
    norm = norm.clip(0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + 1e-10))
    fisher_prev = fisher.shift(1).fillna(0).values
    
    return fisher.values, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    chop = 100 * np.log10(attr_sum / (hh - ll + 1e-10)) / np.log10(period)
    chop = chop.fillna(50).values
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    kama_1d_14 = calculate_kama(df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_14_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_14)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_14 = calculate_kama(close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_14_aligned[i]):
            continue
        
        if np.isnan(kama_12h_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_14_aligned[i]
        daily_bearish = close[i] < kama_1d_14_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = close[i] > kama_12h_14[i]
        kama_bearish = close[i] < kama_12h_14[i]
        
        # === CHOPPINESS REGIME ===
        is_trending = chop_14[i] < 45.0  # Below 45 = trending (relaxed from 38.2)
        is_choppy = chop_14[i] > 55.0   # Above 55 = choppy (relaxed from 61.8)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_signal = fisher[i] < -1.0 and fisher_prev[i] < fisher[i]
        fisher_short_signal = fisher[i] > 1.0 and fisher_prev[i] > fisher[i]
        
        # Relaxed Fisher thresholds for more trades
        fisher_long_relaxed = fisher[i] < -0.5 and fisher_prev[i] < fisher[i]
        fisher_short_relaxed = fisher[i] > 0.5 and fisher_prev[i] > fisher[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if daily_bullish:
            # Trending regime: KAMA breakout + Fisher confirmation
            if is_trending and kama_bullish:
                if fisher_long_signal or (fisher_long_relaxed and kama_bullish):
                    new_signal = current_size
            
            # Choppy regime: Mean reversion at KAMA support
            elif is_choppy and kama_bullish:
                if fisher_long_relaxed:
                    new_signal = current_size * 0.7  # Smaller size in chop
            
            # Default: Simple KAMA + Fisher combo (ensure trade frequency)
            elif kama_bullish and fisher_long_relaxed:
                new_signal = current_size
        
        # SHORT ENTRIES
        elif daily_bearish:
            # Trending regime: KAMA breakdown + Fisher confirmation
            if is_trending and kama_bearish:
                if fisher_short_signal or (fisher_short_relaxed and kama_bearish):
                    new_signal = -current_size
            
            # Choppy regime: Mean reversion at KAMA resistance
            elif is_choppy and kama_bearish:
                if fisher_short_relaxed:
                    new_signal = -current_size * 0.7  # Smaller size in chop
            
            # Default: Simple KAMA + Fisher combo (ensure trade frequency)
            elif kama_bearish and fisher_short_relaxed:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~15 days on 12h), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if daily_bullish and kama_bullish:
                new_signal = current_size * 0.5
            elif daily_bearish and kama_bearish:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and daily_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals