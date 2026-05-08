# USING 1D TIMEFRAME FOR THE STRATEGY
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1D 20-period Donchian breakout with 1W EMA34 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel, price above 1W EMA34, and volume > 2x 20-period average.
# Short when price breaks below lower Donchian channel, price below 1W EMA34, and volume > 2x 20-period average.
# Exit when price crosses back through the Donchian middle line or trend fails.
# Uses tight entry conditions to limit trades (target: 20-40/year) and avoid fee drag.
# Donchian channels provide clear breakout levels; EMA34 filters trend direction.
# Volume spike confirms institutional interest. Designed for 1D timeframe to work in both bull and bear markets.

name = "1D_Donchian_20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (using 20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels based on previous period
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    middle = np.full(len(high_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Align all indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 1d bar's volume
            idx_1d = 0
            while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
                idx_1d += 1
            idx_1d -= 1  # last completed 1d bar
            
            if idx_1d >= 0:
                vol_1d_current = df_1d.iloc[idx_1d]['volume']
                vol_filter = vol_1d_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Price breaks above/below Donchian upper/lower + trend + volume spike
            # Long when price breaks above upper, price above EMA34, with volume spike
            long_condition = (close[i] > upper_aligned[i]) and \
                             ema_34_rising_aligned[i] and (close[i] > ema_34_aligned[i]) and vol_filter
            # Short when price breaks below lower, price below EMA34, with volume spike
            short_condition = (close[i] < lower_aligned[i]) and \
                              ema_34_falling_aligned[i] and (close[i] < ema_34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle or trend fails
            if (close[i] < middle_aligned[i]) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle or trend fails
            if (close[i] > middle_aligned[i]) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals