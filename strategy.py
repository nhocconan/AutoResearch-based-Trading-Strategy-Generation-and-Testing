#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# work well in ranging markets and during pullbacks in trends.
# 1d EMA34 ensures we only take reversals in the direction of the higher timeframe trend.
# Volume confirmation (>1.5x 20-period EMA volume) filters false signals.
# Discrete sizing 0.25 minimizes fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).

name = "6h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 6h data (lookback 14 periods)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll calculate it manually to avoid look-ahead
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.full(n, np.nan)
    mask = hh_ll != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / hh_ll[mask]) * -100
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold reversal) AND 
            # price > 1d EMA34 AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought reversal) AND 
            # price < 1d EMA34 AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) OR 
            # price crosses below 1d EMA34
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) OR 
            # price crosses above 1d EMA34
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals