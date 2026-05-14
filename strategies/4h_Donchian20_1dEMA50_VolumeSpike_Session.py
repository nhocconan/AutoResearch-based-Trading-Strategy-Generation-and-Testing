#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA50 (uptrend) AND 4h volume > 2.0x 20-period volume MA.
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA50 (downtrend) AND 4h volume > 2.0x 20-period volume MA.
# Exit on retracement to Donchian middle band (20-period SMA of high/low) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Donchian channels provide objective price structure, 1d EMA50 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "4h_Donchian20_1dEMA50_VolumeSpike_Session"
timeframe = "4h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from previous 1d bar (OHLC)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20), middle = (upper + lower) / 2
    lookback = 20
    if len(df_1d) < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min for Donchian bands
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donchian_upper = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1).values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_upper_aligned[i]  # Price breaks above upper band
        breakout_down = low_val < donchian_lower_aligned[i]  # Price breaks below lower band
        
        # 1d trend conditions
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
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
            # Exit long: price retouches Donchian middle band OR trend changes
            if close_val < donchian_middle_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian middle band OR trend changes
            if close_val > donchian_middle_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals