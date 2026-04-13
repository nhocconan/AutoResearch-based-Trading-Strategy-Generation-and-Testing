#!/usr/bin/env python3
"""
4h_1d_KC_Breakout_With_Volume
Hypothesis: Keltner Channel breakouts combined with volume expansion and 1-day trend capture strong momentum moves.
Works in bull markets (upward breakouts) and bear markets (downward breakouts) by following the trend with volatility-based channels.
Volume confirms institutional participation. Uses volatility-based stops to manage risk in choppy conditions.
Target: 25-40 trades/year.
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
    
    # Keltner Channel parameters
    atr_period = 10
    kc_multiplier = 2.0
    ema_period = 20
    
    # Calculate True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate ATR using Wilder's smoothing
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
    
    atr = wilder_smooth(tr, atr_period)
    
    # Calculate EMA for Keltner middle line
    ema_middle = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate Keltner Channel bands
    kc_upper = ema_middle + (kc_multiplier * atr)
    kc_lower = ema_middle - (kc_multiplier * atr)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(max(atr_period, ema_period, 20) + 1, n):
        # Skip if any required data is not ready
        if (np.isnan(atr[i]) or np.isnan(ema_middle[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Close breaks above upper Keltner Channel
        # 2. Price above daily EMA50 (1d trend filter)
        # 3. Volume expansion
        breakout_up = close[i] > kc_upper[i-1]
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        long_condition = breakout_up and price_above_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. Close breaks below lower Keltner Channel
        # 2. Price below daily EMA50 (1d trend filter)
        # 3. Volume expansion
        breakout_down = close[i] < kc_lower[i-1]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        short_condition = breakout_down and price_below_ema and volume_expansion[i]
        
        # Exit conditions: reverse position on opposite breakout
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif (position == 1 and close[i] < kc_lower[i-1]) or \
             (position == -1 and close[i] > kc_upper[i-1]):
            # Reverse signal - close position and open opposite
            position = 1 if long_condition else -1
            signals[i] = position_size if position == 1 else -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_KC_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0