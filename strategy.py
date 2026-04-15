#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h trend filter and 1d volume spike
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 4h EMA provides trend direction to filter trades (long in uptrend, short in downtrend).
# 1d volume spike confirms institutional participation and reduces false signals.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Timeframe: 1h, HTF: 4h (trend), 1d (volume)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 1h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 4h EMA (21-period) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average (20-period) for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size: 20% of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + price above 4h EMA (uptrend) + volume spike
        if (williams_r[i] < -80 and 
            close[i] > ema_4h_aligned[i] and
            volume[i] > 1.5 * vol_avg_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + price below 4h EMA (downtrend) + volume spike
        elif (williams_r[i] > -20 and 
              close[i] < ema_4h_aligned[i] and
              volume[i] > 1.5 * vol_avg_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R returns to neutral range (-50) or opposite extreme
        elif position == 1 and williams_r[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_WilliamsR_4hEMA_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0