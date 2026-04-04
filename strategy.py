#!/usr/bin/env python3
"""
exp_6515_6h_elder_ray_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w ADX regime filter. 
In trending markets (ADX>25): trade in direction of Elder Ray + EMA20 filter.
In ranging markets (ADX<20): fade extreme Elder Ray readings near EMA20.
Uses 1w ADX for regime detection to avoid whipsaws. Elder Ray measures bull/bear power 
relative to EMA13. Designed for low-frequency, high-conviction trades with proper 
regime adaptation to work in both bull and bear markets.
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6515_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13
EMA20_PERIOD = 20
ADX_PERIOD = 14
ADX_TRENDING = 25
ADX_RANGING = 20
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for ADX regime
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ADX for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean().values
    
    # Align ADX to LTF (6h) with shift(1) for completed bars only
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=EMA13_PERIOD, min_periods=EMA13_PERIOD, adjust=False).mean().values
    
    # EMA20 for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=EMA20_PERIOD, min_periods=EMA20_PERIOD, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13  # negative values indicate bearish pressure
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(EMA13_PERIOD, EMA20_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]):
            continue
            
        # Regime detection
        is_trending = adx_aligned[i] > ADX_TRENDING
        is_ranging = adx_aligned[i] < ADX_RANGING
        
        # Exit conditions based on regime
        if position == 1:  # long position
            if is_trending:
                # Exit long if bear power becomes strongly negative
                exit_long = bear_power[i] < -np.std(bull_power[max(0,i-50):i+1]) * 1.5
            else:  # ranging
                # Exit long if price crosses below EMA20 or bear power turns negative
                exit_long = close[i] < ema20[i] or bear_power[i] > 0
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if is_trending:
                # Exit short if bull power becomes strongly positive
                exit_short = bull_power[i] > np.std(bear_power[max(0,i-50):i+1]) * 1.5
            else:  # ranging
                # Exit short if price crosses above EMA20 or bull power turns negative
                exit_short = close[i] > ema20[i] or bull_power[i] < 0
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if is_trending:
                # Trending regime: trade with Elder Ray direction + EMA20 filter
                if bull_power[i] > 0 and close[i] > ema20[i]:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                elif bear_power[i] < 0 and close[i] < ema20[i]:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: fade extreme Elder Ray near EMA20
                if bull_power[i] < -np.std(bull_power[max(0,i-100):i+1]) * 2.0 and close[i] < ema20[i]:
                    # Extremely bearish power but price below EMA20 -> long (mean reversion)
                    signals[i] = SIGNAL_SIZE
                    position = 1
                elif bear_power[i] > np.std(bear_power[max(0,i-100):i+1]) * 2.0 and close[i] > ema20[i]:
                    # Extremely bullish power but price above EMA20 -> short (mean reversion)
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: stay flat
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals