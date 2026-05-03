#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 1w EMA50 is rising AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) lower band AND 1w EMA50 is falling AND volume > 1.5x 20-period MA.
# Exit when price returns to Donchian(20) midpoint OR volume drops below average.
# Uses 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Donchian channels provide clear breakout levels, 1w EMA50 filters for higher-timeframe trend,
# volume confirmation ensures institutional participation. Works in bull (breakouts with trend) and
# bear (breakdowns with trend) markets by trading in the direction of the 1w trend.

name = "1d_DonchianBreakout_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = np.nan
    ema_50_rising = ema_50_1w > ema_50_1w_prev
    ema_50_falling = ema_50_1w < ema_50_1w_prev
    
    # Align 1w indicators to 1d timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 1w EMA50 rising AND volume spike AND session
            if close[i] > highest_high[i] and ema_50_rising_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 1w EMA50 falling AND volume spike AND session
            elif close[i] < lowest_low[i] and ema_50_falling_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR volume drops below average
            if close[i] <= donchian_mid[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR volume drops below average
            if close[i] >= donchian_mid[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals