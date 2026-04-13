#!/usr/bin/env python3
"""
4h_1d_Vortex_Volume_Trend_Strategy
Hypothesis: The Vortex Indicator (VI) identifies trend direction, with VI+ > VI- indicating uptrend and vice versa.
Combined with 1-day trend confirmation (EMA50) and volume expansion, this captures strong trends while avoiding whipsaws.
Volume expansion confirms institutional participation. Works in both bull (strong uptrends) and bear (strong downtrends) markets.
Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Vortex Indicator on 4h data
    # True Range components
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original indices
    
    # Vortex components
    vm_plus = np.abs(high[1:] - low[:-1])  # |high - prev_low|
    vm_minus = np.abs(low[1:] - high[:-1])  # |low - prev_high|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) if np.any(np.isnan(data[1:period])) else np.sum(data[1:period]) / period
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    vi_plus = wilder_smooth(vm_plus, 14)
    vi_minus = wilder_smooth(vm_minus, 14)
    vt_sum = wilder_smooth(tr, 14)
    
    # Avoid division by zero
    vi_plus = np.where(vt_sum != 0, vi_plus / vt_sum, 0)
    vi_minus = np.where(vt_sum != 0, vi_minus / vt_sum, 0)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. VI+ > VI- (uptrend signal)
        # 2. Price above daily EMA50 (1d trend filter)
        # 3. Volume expansion
        vi_bullish = vi_plus[i] > vi_minus[i]
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        long_condition = vi_bullish and price_above_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. VI- > VI+ (downtrend signal)
        # 2. Price below daily EMA50 (1d trend filter)
        # 3. Volume expansion
        vi_bearish = vi_minus[i] > vi_plus[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        short_condition = vi_bearish and price_below_ema and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Vortex_Volume_Trend_Strategy"
timeframe = "4h"
leverage = 1.0