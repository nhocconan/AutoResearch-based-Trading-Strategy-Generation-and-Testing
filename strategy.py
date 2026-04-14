#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout with 1d trend filter and volume confirmation
# Uses 1d EMA200 for trend direction (works in bull/bear via long/short symmetry)
# 1h Donchian(20) breakout for entry timing with volume > 1.5x average
# Time-based filter: 08-20 UTC to avoid low-volume Asian session
# Target: 15-30 trades/year per symbol (60-120 over 4 years)
# Position size: 0.20 (20%) to control drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 1h Donchian channels (20 periods)
    donch_len = 20
    upper_channel = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_channel = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(100, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        above_ema = close[i] > ema_200_aligned[i]
        below_ema = close[i] < ema_200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + above EMA200 + volume + session
            if (close[i] > upper_channel[i-1] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakout down + below EMA200 + volume + session
            elif (close[i] < lower_channel[i-1] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian lower channel (stop and reverse)
            if close[i] < lower_channel[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian upper channel (stop and reverse)
            if close[i] > upper_channel[i]:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_EMA200_Donchian_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0