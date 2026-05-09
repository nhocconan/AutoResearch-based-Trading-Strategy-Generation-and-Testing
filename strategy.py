#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions. In trending markets (1d EMA34),
# we take counter-trend entries at extreme %R levels with volume confirmation.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
# Target: 15-30 trades/year to minimize fee drag.
name = "12h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA for daily timeframe
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams %R (14-period) for 12h timeframe
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # For true Williams %R, we need to reset the accumulation every 14 periods
    williams_r = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i < 14:
            williams_r[i] = np.nan
        else:
            highest_high_14 = np.max(high[i-13:i+1])
            lowest_low_14 = np.min(low[i-13:i+1])
            if highest_high_14 - lowest_low_14 != 0:
                williams_r[i] = (highest_high_14 - close[i]) / (highest_high_14 - lowest_low_14) * -100
            else:
                williams_r[i] = -50  # neutral when no range
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 34 periods for EMA and 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price above EMA34 (uptrend) + Williams %R oversold (< -80) + volume spike
            if (price > ema_34_1d_aligned[i] and williams_r[i] < -80 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA34 (downtrend) + Williams %R overbought (> -20) + volume spike
            elif (price < ema_34_1d_aligned[i] and williams_r[i] > -20 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34 or Williams %R returns above -50
            if price < ema_34_1d_aligned[i] or williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 or Williams %R returns below -50
            if price > ema_34_1d_aligned[i] or williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals