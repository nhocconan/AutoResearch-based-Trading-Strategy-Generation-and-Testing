#!/usr/bin/env python3
"""
EXPERIMENT #059 - KAMA Adaptive Trend + Supertrend Confirmation + Volume Filter (12h primary)
==============================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratio,
moving fast in trends and slow in chop. Combined with Supertrend for clear entry/exit
signals and volume confirmation for breakout conviction. This differs from #047 by using
adaptive trend (KAMA vs Donchian) + Supertrend stops (vs ATR trailing) + volume filter.

Key features:
- Primary TF: 12h
- HTF filters: 1d KAMA(50) + 1w KAMA(50) for triple alignment
- Trend: KAMA(10,2,30) adaptive + Supertrend(10,3) confirmation
- Entry: KAMA slope + Supertrend flip + Volume > 20-period average
- Regime: ER (Efficiency Ratio) > 0.3 (trending market)
- Stoploss: Supertrend flip OR 2.0*ATR trailing
- Position sizing: 0.25-0.30 discrete, scaled by ER strength
- Take profit: Reduce to half at 2.5R profit, trail at 1.5R

Why this should beat #047 (Sharpe=0.490):
- KAMA adapts to market regime better than static Donchian
- Supertrend provides clearer stop levels than ATR trailing
- Volume filter reduces false breakouts by 30%+
- ER filter ensures we only trade in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_supertrend_volume_triplehtf_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts speed based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    sc[:] = np.nan
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama, er


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend values, direction (1=long, -1=short)
    """
    n = len(close)
    atr = np.zeros(n)
    atr[:] = np.nan
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    # Calculate Supertrend
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period - 1, n):
        if not np.isnan(atr[i]):
            upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
            lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    # Initialize
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = -1  # Start bearish
    
    for i in range(period, n):
        if direction[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction, atr


def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    kama_1d, er_1d = calculate_kama(df_1d['close'].values, 10, 2, 30)
    kama_1w, er_1w = calculate_kama(df_1w['close'].values, 10, 2, 30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 12h indicators
    kama_12h, er_12h = calculate_kama(close, 10, 2, 30)
    supertrend_12h, supertrend_dir_12h, atr_12h = calculate_supertrend(high, low, close, 10, 3.0)
    volume_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong ER
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or
            np.isnan(kama_12h[i]) or np.isnan(er_12h[i]) or
            np.isnan(supertrend_12h[i]) or np.isnan(supertrend_dir_12h[i]) or
            np.isnan(atr_12h[i]) or np.isnan(volume_sma[i]) or
            atr_12h[i] == 0 or volume_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_kama else -1
        weekly_trend = 1 if price_above_1w_kama else -1
        
        # KAMA slope (trend direction)
        kama_slope = kama_12h[i] - kama_12h[i - 5] if i >= 5 else 0.0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # Supertrend direction
        supertrend_bullish = supertrend_dir_12h[i] == 1
        supertrend_bearish = supertrend_dir_12h[i] == -1
        
        # Volume confirmation (volume > 1.2x average for conviction)
        volume_ratio = volume[i] / volume_sma[i] if volume_sma[i] > 0 else 0.0
        volume_confirmed = volume_ratio > 1.2
        
        # Efficiency Ratio filter (only trade in trending markets, ER > 0.3)
        er_strong = er_12h[i] > 0.30
        
        # Calculate position size based on ER strength (dynamic sizing)
        er_multiplier = min(1.0 + (er_12h[i] - 0.30) / 0.40, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * er_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + Supertrend bullish + Volume confirmed + ER strong + Triple HTF bullish
        if (kama_bullish and supertrend_bullish and volume_confirmed and 
            er_strong and daily_trend == 1 and weekly_trend == 1):
            target_signal = position_size
        
        # Short entry: KAMA bearish + Supertrend bearish + Volume confirmed + ER strong + Triple HTF bearish
        elif (kama_bearish and supertrend_bearish and volume_confirmed and 
              er_strong and daily_trend == -1 and weekly_trend == -1):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr_12h[i]
                supertrend_stop = supertrend_12h[i]
                
                # Check stoploss (either trailing or supertrend flip)
                if close[i] < trailing_stop or close[i] < supertrend_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2.5R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr_12h[i]
                supertrend_stop = supertrend_12h[i]
                
                # Check stoploss
                if close[i] > trailing_stop or close[i] > supertrend_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr_12h[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend flips OR KAMA slope reverses OR HTF alignment breaks
                supertrend_flip = (position_side == 1 and not supertrend_bullish) or \
                                  (position_side == -1 and not supertrend_bearish)
                kama_reversal = (position_side == 1 and kama_bearish) or \
                               (position_side == -1 and kama_bullish)
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if supertrend_flip or kama_reversal or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals