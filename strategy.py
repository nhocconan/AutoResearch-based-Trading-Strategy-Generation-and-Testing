# 1h_Donchian20_Breakout_4hTrend_VolumeSmooth
# Hypothesis: Donchian(20) breakouts on 1h capture momentum with 4h EMA20 trend filter.
# Uses volume smoothing (ratio > 1.5) and session filter (08-20 UTC) to reduce false signals.
# Targets 15-30 trades/year by requiring confluence of breakout, trend, volume, and session.
# Works in bull/bear via trend filter; volume smooth avoids whipsaws.
# Position size 0.20 balances risk and return.

name = "1h_Donchian20_Breakout_4hTrend_VolumeSmooth"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime64 issues
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Donchian(20) channels on 1h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above 20-period high or below 20-period low
        breakout_up = close[i] > high_20[i-1]  # Use previous bar's high to avoid look-ahead
        breakout_down = close[i] < low_20[i-1]  # Use previous bar's low
        
        # Volume confirmation: volume > 1.5x average (smoothed)
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter from 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend + session
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: downward breakout + volume + downtrend + session
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks back below 20-period low or trend reversal
            if close[i] < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks back above 20-period high or trend reversal
            if close[i] > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals