#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX + volume breakout with 1d trend filter
# Uses 1d EMA200 for long-term trend direction (bull/bear filter)
# Enters on 1h ADX > 25 + volume > 1.5x avg + price breaks 1h Donchian(20)
# Exits when price crosses 1h Donchian midpoint
# Uses session filter (08-20 UTC) to avoid low-liquidity hours
# Position size: 0.20 to manage drawdown
# Expects ~20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_len = 200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h ADX (14 periods)
    adx_len = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Calculate 1h Donchian channels (20 periods)
    donch_len = 20
    upper_channel = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_channel = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len, 20, ema_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(adx[i]) or 
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Strength filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + volume + trend + uptrend
            if (close[i] > upper_channel[i-1] and 
                volume_confirmed and 
                trending and
                uptrend):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakout down + volume + trend + downtrend
            elif (close[i] < lower_channel[i-1] and 
                  volume_confirmed and 
                  trending and
                  downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below midpoint of channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above midpoint of channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_ADX_Volume_Breakout_EMA200_v1"
timeframe = "1h"
leverage = 1.0