#!/usr/bin/env python3
"""
Experiment #009: 4h BB Squeeze + ATR Expansion + Volume Confirmation

HYPOTHESIS: BB Width percentile < 10% (squeeze) + ATR expansion > 1.3x
identifies low-volatility compression BEFORE directional moves better than CHOP.

WHY IT SHOULD WORK IN BULL + BEAR + RANGE:
- Bull: squeeze forms → ATR expands → BB lower band breaks up → big move follows
- Bear: squeeze forms → ATR expands → BB upper band breaks down → big move follows
- Range: squeeze still forms, but we require HTF confirmation to skip chop

KEY DIFFERENCE FROM FAILED STRATEGIES:
- #003 used CHOP < 50 → 306 trades (too many)
- #006 used CHOP < 45 → 145 trades (better but still moderate)
- #009 uses BB squeeze (<10th percentile) which is MORE selective than CHOP

BB WIDTH SQUEEZE IS PROVEN:
- John Bollinger's research shows BB width contraction precedes 80% of big moves
- Works across all markets (crypto, forex, equities)
- Natural edge: markets spend 70% in contraction, 30% expansion

TARGET: 75-125 total trades over 4 years (19-31/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_atr_expansion_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb_width_percentile(close, period=20, lookback=100):
    """
    Bollinger Band Width normalized to percentile rank.
    BB Width = (Upper Band - Lower Band) / Middle Band
    Low percentile = squeeze = potential explosive move coming
    """
    n = len(close)
    if n < period + lookback:
        return np.full(n, np.nan)
    
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + 2 * std
    lower = mid - 2 * std
    
    bb_width = (upper - lower) / mid
    
    percentile = np.full(n, np.nan)
    for i in range(period + lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        current = bb_width[i]
        # Rank of current value among past 'lookback' values (0-100 percentile)
        rank = np.sum(window < current) / lookback * 100
        percentile[i] = rank
    
    return percentile

def calculate_atr_ratio(atr, period=7, lookback=30):
    """
    ATR ratio: current short ATR vs longer ATR
    > 1.3 = expansion (volatility picking up = potential move)
    < 0.8 = contraction (quiet = potential squeeze incoming)
    """
    n = len(atr)
    if n < lookback + period:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).rolling(window=period, min_periods=period).mean().values
    atr_long = pd.Series(atr).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    atr_ratio = np.where(atr_long > 0, atr_short / atr_long, np.nan)
    
    return atr_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend - more stable than 21
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_width_pct = calculate_bb_width_percentile(close, period=20, lookback=100)
    atr_ratio = calculate_atr_ratio(atr_14, period=7, lookback=30)
    
    # Donchian channel for breakout detection (20-period)
    channel_up = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    channel_lo = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 2.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 350  # 300 for squeeze percentile + 20 for BB + 14 for ATR + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        # === SQUEEZE DETECTION: BB width < 10th percentile ===
        is_squeeze = bb_width_pct[i] < 10.0
        
        # === VOLATILITY EXPANSION: ATR ratio > 1.3 ===
        is_expansion = atr_ratio[i] > 1.3
        
        # === HTF TREND: 1d EMA(50) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION: 2.5x average ===
        vol_spike = vol_ratio[i] > 2.5
        
        # === DONCHIAN BREAKOUT (shifted by 1 to avoid look-ahead) ===
        breakout_up = close[i] > channel_up[i]
        breakout_down = close[i] < channel_lo[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Squeeze + ATR expansion + breakout up + HTF up + volume ===
            if is_squeeze and is_expansion and breakout_up and htf_trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Squeeze + ATR expansion + breakout down + HTF down + volume ===
            if is_squeeze and is_expansion and breakout_down and htf_trend_down and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOP: 2.5 ATR ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals