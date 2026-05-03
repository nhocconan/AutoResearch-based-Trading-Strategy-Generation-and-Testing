#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above 1d Donchian upper channel AND 1w close > 1w EMA34 (uptrend) AND 1d volume > 2.0x 20-period volume MA.
# Short when price breaks below 1d Donchian lower channel AND 1w close < 1w EMA34 (downtrend) AND 1d volume > 2.0x 20-period volume MA.
# Exit on retracement to 1d Donchian midpoint or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Donchian channels provide objective breakout levels, 1w EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "1d_Donchian20_1wEMA34_VolumeSpike_Session"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    if len(prices) < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min for Donchian channels
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_1d[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1d volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_1d[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > highest_high[i]  # Price breaks above upper channel
        breakout_down = low_val < lowest_low[i]   # Price breaks below lower channel
        
        # 1w trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 1w uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 1w downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint OR trend changes
            if close_val < donchian_mid[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian midpoint OR trend changes
            if close_val > donchian_mid[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals