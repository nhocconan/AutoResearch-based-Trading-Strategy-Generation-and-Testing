#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversal with 1w EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. In 12h timeframe, readings below -80 
# indicate oversold (long opportunity) and above -20 indicate overbought (short opportunity).
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike (>1.8x 24 EMA) confirms participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "12h_WilliamsR_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (completed 1w bar only)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R(14) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: 24-period EMA of volume on 12h timeframe
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 24-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_24[i])
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) + uptrend + volume spike
            if williams_r[i] < -80.0 and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + downtrend + volume spike
            elif williams_r[i] > -20.0 and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (moving out of oversold) OR trend changes OR volume drops
            if (williams_r[i] > -50.0 or 
                close[i] < ema50_1w_aligned[i] or 
                volume[i] < vol_ema_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (moving out of overbought) OR trend changes OR volume drops
            if (williams_r[i] < -50.0 or 
                close[i] > ema50_1w_aligned[i] or 
                volume[i] < vol_ema_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals