# 6h_PriceChannel_VolumeBreakout_v1
# Hypothesis: Price channel breakouts on 6h with volume confirmation and 1-day trend filter.
# Uses Donchian channels (20-period) as dynamic support/resistance. Enters on breakouts
# with volume > 1.5x 20-period average only when aligned with 1-day EMA trend.
# Exits on channel middle reversion. Designed for 15-35 trades/year to minimize fee drag.
# Works in bull/bear via trend filter - only takes breakouts in direction of higher timeframe.

name = "6h_PriceChannel_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) - price channel structure
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    channel_mid = (high_roll + low_roll) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-day trend filter: EMA of daily close (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(channel_mid[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper channel AND volume spike AND above 1-day EMA
            if (close[i] > high_roll[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower channel AND volume spike AND below 1-day EMA
            elif (close[i] < low_roll[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below channel middle (mean reversion)
            if close[i] < channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above channel middle (mean reversion)
            if close[i] > channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals