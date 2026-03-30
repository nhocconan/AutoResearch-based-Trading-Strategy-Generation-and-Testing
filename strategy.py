#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian + 1d Trend + Volume Spike + Chop Regime

HYPOTHESIS:
Based on proven DB winner mtf_4h_chop_donchian_vol_regime_12h_v1 (test Sharpe 1.491, 107 trades),
but adapted for 12h primary timeframe with 1d HTF reference.

WHY THIS SHOULD WORK:
1. 12h TF with 54% keep rate (experiment data) - better than 4h for this session
2. Donchian(30) breakout on 12h - structural levels with enough history
3. 1d HMA(21) for HTF trend - aligns with proven HTF patterns in winners
4. Volume spike 1.8x confirmation - proven volume filter from session best
5. CHOP<50 for trending regime - filters out range markets
6. Target 75-150 total trades over 4 years (19-37/year) - within optimal range

ENTRY PHILOSOPHY: "Wait for the channel to compress, then explode"
- Donchian narrowing = energy building
- Volume spike on breakout = confirmation
- HTF trend alignment = higher probability

STOPLOSS: 2.5x ATR trailing - proven risk management from winners
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_vol_chop_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - AVOID
    CHOP < 50 = trending - ENTER
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=30):
    """Donchian Channel - higher period for 12h TF"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume spike detection"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for HTF trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position - proven size from session best
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 300  # 30 donchian + 14 CHOP + 20 vol MA + HTF alignment buffer
    
    for i in range(warmup, n):
        # === Validations ===
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8  # Don't enter in range
        is_trending = chop_value < 50   # Good to enter in trend
        
        # === HTF trend from 1d HMA ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === Volume confirmation ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === Donchian breakout detection ===
        prev_up = donchian_up[i - 1]
        prev_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_up
        breakout_down = close[i] < prev_lo
        
        # === Entry logic ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + HTF up + volume ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Trending + breakout down + HTF down + volume ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS: 2.5x ATR trailing stop ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop if price falls 2.5 ATR from high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit on HTF trend flip
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if range market
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop if price rises 2.5 ATR from low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit on HTF trend flip
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if range market
                if is_choppy:
                    desired_signal = 0.0
        
        # === Minimum hold: 3 bars (12h = 36h min) ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === Position management ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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