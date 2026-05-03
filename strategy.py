#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 12h Donchian upper band AND 1d close > 1d EMA50 (uptrend) AND 12h volume > 2.0x 20-period volume MA.
# Short when price breaks below 12h Donchian lower band AND 1d close < 1d EMA50 (downtrend) AND 12h volume > 2.0x 20-period volume MA.
# Exit on retracement to midpoint of Donchian channel or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channel provides clear breakout levels, 1d EMA50 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "12h_Donchian20_1dEMA50_VolumeSpike_Session"
timeframe = "12h"
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
    
    # Calculate 12h Donchian channel (20-period) from previous 12h data to avoid look-ahead
    # We need to shift the 12h data by 1 to avoid look-ahead (use previous 12h bar's levels)
    # Since we don't have direct 12h data, we'll calculate from prices but shift appropriately
    high_12h = pd.Series(high).shift(20)  # Shift by 20 to simulate previous 12h bar for 20-period lookback
    low_12h = pd.Series(low).shift(20)
    # Actually, we need to calculate Donchian on the 12h timeframe properly
    # Let's resample conceptually but using mtf_data approach - we'll use 12h data from get_htf_data
    # But since timeframe is 12h, we need to get 12h data for Donchian calculation
    # However, we can't call get_htf_data for same timeframe, so we'll calculate on prices but ensure no look-ahead
    # Correct approach: calculate Donchian on 12h data using historical values only
    # We'll use a rolling window but ensure we only use completed periods
    
    # Calculate 20-period Donchian channel using only historical data (no look-ahead)
    # For bar i, we use data from 0 to i-1 to calculate the channel for bar i
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        # Look back 20 periods from i-1 to avoid look-ahead
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 12h volume 20-period MA for spike detection (using historical data only)
    volume_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma_12h[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_12h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 12h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_upper[i]  # Price breaks above upper band
        breakout_down = low_val < donchian_lower[i]  # Price breaks below lower band
        
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
            # Exit long: price retouches midpoint of Donchian channel OR trend changes
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if close_val < midpoint or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint of Donchian channel OR trend changes
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if close_val > midpoint or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals