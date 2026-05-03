#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above 6h Donchian upper band AND 1d close > 1d EMA34 (uptrend) AND 6h volume > 2.0x 20-period volume MA.
# Short when price breaks below 6h Donchian lower band AND 1d close < 1d EMA34 (downtrend) AND 6h volume > 2.0x 20-period volume MA.
# Exit on retracement to 6h Donchian mid-band (mean of upper/lower) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear breakout levels, 1d EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "6h_Donchian20_1dEMA34_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll
    lower_band = low_roll
    mid_band = (upper_band + lower_band) / 2.0
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(mid_band[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = close_val > upper_band[i]   # Close breaks above upper band
        breakout_down = close_val < lower_band[i]  # Close breaks below lower band
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 1d uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 1d downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian mid-band OR trend changes
            if close_val < mid_band[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian mid-band OR trend changes
            if close_val > mid_band[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals