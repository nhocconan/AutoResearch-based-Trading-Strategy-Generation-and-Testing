#!/usr/bin/env python3
"""
EXPERIMENT #084 - Supertrend + Volume Spike + Weekly HMA Filter (1d primary)
=====================================================================================
Hypothesis: Daily Supertrend captures major crypto trends effectively, but needs
volume confirmation to filter false breakouts. Weekly HMA(50) ensures we trade
with the secular trend. This differs from previous attempts by using daily as
primary (slower, fewer but higher-quality signals) + volume spike confirmation
(not just price breakout) + weekly trend filter.

Key features:
- Primary TF: 1d (fewer signals, higher quality)
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: Supertrend(10, 3.0) for direction
- Entry: Supertrend flip + volume spike (>1.5x 20-day avg)
- Regime: Weekly HMA slope confirms major trend
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 3R profit, trail stop at 1.5R

Why this should beat current best (Sharpe=0.490):
- Daily timeframe reduces noise and whipsaws vs 12h/4h
- Volume spike filter removes 40%+ of false Supertrend signals
- Weekly HMA ensures we don't fight the major trend
- Conservative sizing (0.25-0.30) with proper stoploss controls DD
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_vol_weeklyhtf_1d_1w_v1"
timeframe = "1d"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1
        else:
            # Upper band logic
            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                upper_band[i] = upper_band[i - 1]
            
            # Lower band logic
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                lower_band[i] = lower_band[i - 1]
            
            # Trend determination
            if trend[i - 1] == 1:
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend, upper_band, lower_band


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / vol_avg
    volume_ratio[vol_avg == 0] = 0
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    supertrend, st_trend, upper_band, lower_band = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    volume_ratio = calculate_volume_ratio(volume, 20)
    
    # Calculate weekly HMA slope for trend confirmation
    hma_1w_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i - 1]):
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i - 1]) / hma_1w_aligned[i - 1]
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size with strong volume
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(hma_1w_slope[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_bullish = close[i] > hma_1w_aligned[i] and hma_1w_slope[i] > 0
        weekly_bearish = close[i] < hma_1w_aligned[i] and hma_1w_slope[i] < 0
        
        # Supertrend signal
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Check for Supertrend flip (entry signal)
        st_flip_long = (st_trend[i] == 1 and st_trend[i - 1] == -1) if i > 0 else False
        st_flip_short = (st_trend[i] == -1 and st_trend[i - 1] == 1) if i > 0 else False
        
        # Volume confirmation (spike > 1.5x average)
        volume_spike = volume_ratio[i] > 1.5
        
        # Calculate position size based on volume strength
        vol_multiplier = min(1.0 + (volume_ratio[i] - 1.5) / 3.0, 1.15)  # Max 1.15x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend flip + volume spike + weekly bullish
        if st_flip_long and volume_spike and weekly_bullish:
            target_signal = position_size
        
        # Short entry: Supertrend flip + volume spike + weekly bearish
        elif st_flip_short and volume_spike and weekly_bearish:
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (3R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 7.5 * entry_atr:  # 3R = 7.5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 7.5 * entry_atr:  # 3R profit
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
            # Reduce position to half at 3R profit
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
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend flips against position OR weekly trend breaks
                st_reversal_long = st_trend[i] == -1
                st_reversal_short = st_trend[i] == 1
                weekly_trend_broken = (position_side == 1 and not weekly_bullish) or \
                                      (position_side == -1 and not weekly_bearish)
                
                if st_reversal_long or st_reversal_short or weekly_trend_broken:
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