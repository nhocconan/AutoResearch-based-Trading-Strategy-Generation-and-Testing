#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Volume Spike + 1w Donchian Breakout with ADX trend filter
# Volume spikes confirm institutional interest; breakouts from weekly channels capture momentum
# ADX filter ensures trades only in trending markets, reducing false breakouts in ranges
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation)
# Target: 20-40 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Volume Spike detection
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Volume Spike: current volume > 2x 20-day average volume
    vol_20_avg = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (2.0 * vol_20_avg)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Load 1w data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian channels (25 periods for less sensitivity)
    donch_length = 25
    donch_high = pd.Series(df_1w['high']).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Align Donchian channels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Calculate ADX on 1d for trend strength (14-period)
    # Need +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = prev_smooth - (prev_smooth/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # DI values
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for all indicators
    start = max(50, 50)  # Need enough for calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vol_spike_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Breakout signals from 1w Donchian
        breakout_up = price > donch_high_aligned[i]
        breakout_down = price < donch_low_aligned[i]
        
        if position == 0:
            # Enter long: volume spike + upward breakout + trending market
            if vol_spike_aligned[i] and breakout_up and trending:
                position = 1
                signals[i] = position_size
            # Enter short: volume spike + downward breakout + trending market
            elif vol_spike_aligned[i] and breakout_down and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR ADX drops below 20 (trend weakening)
            if price < donch_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR ADX drops below 20
            if price > donch_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dVolSpike_1wDonchian_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0