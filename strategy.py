#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
# - Uses Donchian channel (20) breakouts for entry
# - Volume spike (>2x 20-bar average) confirms breakout strength
# - ADX(14) > 25 ensures trending market to avoid false breakouts in chop
# - Works in bull/bear by only taking breakouts in direction of ADX trend
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_DonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakout
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # ADX(14) for trend strength filter
    # Calculate +DM, -DM, TR
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    
    up_move = high - high_shift
    down_move = low_shift - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    tr_smoothed = wilder_smooth(tr, 14)
    plus_dm_smoothed = wilder_smooth(plus_dm, 14)
    minus_dm_smoothed = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and ADX > 25
            long_cond = (close[i] > high_max[i] and 
                        volume_spike[i] and
                        adx[i] > 25)
            
            # Short: price breaks below Donchian low with volume spike and ADX > 25
            short_cond = (close[i] < low_min[i] and 
                         volume_spike[i] and
                         adx[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals