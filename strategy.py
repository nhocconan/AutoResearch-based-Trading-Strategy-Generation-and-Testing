#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper band AND 1d close > 1d EMA50 (uptrend) AND 12h volume > 1.5x 20-period volume MA.
# Short when price breaks below 12h Donchian lower band AND 1d close < 1d EMA50 (downtrend) AND 12h volume > 1.5x 20-period volume MA.
# Exit on retracement to opposite Donchian band or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear trend structure, 1d EMA50 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "12h_Donchian20_1dEMA50_VolumeConfirmation_Session"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period) from previous close to avoid look-ahead
    # Using rolling window on 12h data, but we need to shift by 1 to avoid look-ahead
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    donchian_upper = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume 20-period MA for volume confirmation
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        # Volume confirmation: current 12h volume > 1.5x 20-period volume MA
        volume_confirm = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_upper[i]   # Price breaks above upper band
        breakout_down = low_val < donchian_lower[i]  # Price breaks below lower band
        
        # 1d trend conditions
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 1d uptrend AND volume confirmation AND session
            if breakout_up and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 1d downtrend AND volume confirmation AND session
            elif breakout_down and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches lower band OR trend changes
            if close_val < donchian_lower[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches upper band OR trend changes
            if close_val > donchian_upper[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals