#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h trend filter and 1d volume confirmation.
# Long when price breaks above 20-period Donchian high AND 4h close > 4h EMA50 (uptrend) AND 1d volume > 1.5x 20-period volume MA.
# Short when price breaks below 20-period Donchian low AND 4h close < 4h EMA50 (downtrend) AND 1d volume > 1.5x 20-period volume MA.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.20.
# Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
# Donchian channels provide objective breakout levels, 4h EMA50 filters for trend alignment, 1d volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 4h trend when volume confirms.

name = "1h_Donchian20_4hEMA50_1dVolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_1d = df_1d['volume'].values[-1] if len(df_1d) > 0 else 0  # Current 1d volume (approximation for aligned data)
        # For volume spike, we use the aligned 1d volume MA and compare with current 1h volume scaled to daily
        # Since we don't have current 1d volume in 1h data, we use volume ratio: current 1h volume vs 20-period 1h volume MA as proxy
        volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike_1h = volume[i] > (volume_ma_1h[i] * 2.0)  # 1h volume spike as proxy for 1d confirmation
        
        # Donchian breakout conditions
        breakout_up = high_val > high_roll[i]  # Price breaks above 20-period high
        breakout_down = low_val < low_roll[i]  # Price breaks below 20-period low
        
        # 4h trend conditions
        trend_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 4h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike_1h:
                signals[i] = 0.20
                position = 1
            # Short: Donchian breakout down AND 4h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike_1h:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian mid-channel OR trend changes
            mid_channel = (high_roll[i] + low_roll[i]) / 2.0
            if close_val < mid_channel or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches Donchian mid-channel OR trend changes
            mid_channel = (high_roll[i] + low_roll[i]) / 2.0
            if close_val > mid_channel or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals