#!/usr/bin/env python3
# 6h_adx_donchian_1w_trend_volume_v1
# Hypothesis: Combine ADX trend strength with Donchian breakouts and weekly trend filter.
# Enter long when price breaks above Donchian(20) high + ADX > 25 + weekly trend up.
# Enter short when price breaks below Donchian(20) low + ADX > 25 + weekly trend down.
# Uses volume confirmation to avoid false breakouts.
# Designed for 6h timeframe to target 50-150 trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_donchian_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth(values, period):
            smoothed = np.zeros_like(values)
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed
        
        tr_sum = smooth(tr, period)
        plus_dm_sum = smooth(plus_dm, period)
        minus_dm_sum = smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_sum > 0, 100 * plus_dm_sum / tr_sum, 0)
        minus_di = np.where(tr_sum > 0, 100 * minus_dm_sum / tr_sum, 0)
        
        dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
            else:
                upper[i] = np.nan
                lower[i] = np.nan
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma[i] = np.nan
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(adx[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian lower OR weekly trend turns down
            if (close[i] < donch_lower[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian upper OR weekly trend turns up
            if (close[i] > donch_upper[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > Donchian upper + ADX > 25 + volume + weekly trend up
            if (close[i] > donch_upper[i]) and (adx[i] > 25) and volume_filter[i] and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Donchian lower + ADX > 25 + volume + weekly trend down
            elif (close[i] < donch_lower[i]) and (adx[i] > 25) and volume_filter[i] and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals