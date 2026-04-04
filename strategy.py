#!/usr/bin/env python3
"""
exp_6571_6h_elder_ray_1d_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX + EMA200).
Elder Ray measures bull/bear power relative to EMA13. In trending regimes (ADX>25, price>EMA200),
we take trend-following signals. In ranging regimes (ADX<20), we fade extremes.
Uses 6h primary timeframe to target 50-150 total trades over 4 years.
Discrete sizing (0.25) minimizes fee churn. Works in both bull and bear markets via regime adaptation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6571_6h_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13      # For Elder Ray calculation
EMA200_PERIOD = 200    # For regime filter (1d)
ADX_PERIOD = 14        # For regime filter (1d)
SIGNAL_SIZE = 0.25     # 25% position size
ADX_TREND_THRESHOLD = 25   # Above = trending
ADX_RANGE_THRESHOLD = 20   # Below = ranging
MAX_HOLD_BARS = 24     # Max hold: ~6 days (4h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=EMA200_PERIOD, min_periods=EMA200_PERIOD, adjust=False).mean().values
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+
    tr_period = pd.Series(tr).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(tr_period == 0, np.nan, tr_period)
    di_minus = 100 * dm_minus_smooth / np.where(tr_period == 0, np.nan, tr_period)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx_1d = pd.Series(dx).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    
    # Align HTF regime filters to LTF (6h) with shift(1) for completed bars only
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=EMA13_PERIOD, min_periods=EMA13_PERIOD, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA13_PERIOD, EMA200_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Determine regime: trending or ranging
        is_trending = adx_1d_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = adx_1d_aligned[i] < ADX_RANGE_THRESHOLD
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        
        # Exit conditions: time-based exit OR Elder Ray reversal
        if position == 1:  # long position
            # Exit if Bear Power turns positive (bulls losing control)
            exit_long = bear_power[i] > 0
            # Time-based exit
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if Bull Power turns negative (bears losing control)
            exit_short = bull_power[i] < 0
            # Time-based exit
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if is_trending:
                # Trending regime: follow Elder Ray with trend filter
                if bull_power[i] > 0 and price_above_ema200:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power[i] < 0 and not price_above_ema200:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: fade extremes (contrarian)
                if bull_power[i] < 0 and bear_power[i] < 0:  # Both weak - mean reversion long
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bull_power[i] > 0 and bear_power[i] > 0:  # Both strong - mean reversion short
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no clear signal
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals