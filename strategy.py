#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Williams %R: measures overbought/oversold levels (-20 to -80)
# Long when Williams %R crosses above -80 from below AND price > 1d EMA34 AND volume spike
# Short when Williams %R crosses below -20 from above AND price < 1d EMA34 AND volume spike
# Uses mean reversion in trends with trend filter to avoid counter-trend whipsaws
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h (14-period)
    if len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R crossovers
        if i > 50:
            williams_r_prev = williams_r[i-1]
            williams_r_curr = williams_r[i]
            # Long: crosses above -80 from below
            long_signal = (williams_r_prev <= -80) and (williams_r_curr > -80)
            # Short: crosses below -20 from above
            short_signal = (williams_r_prev >= -20) and (williams_r_curr < -20)
        else:
            long_signal = False
            short_signal = False
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND price > 1d EMA34 AND volume spike
            if long_signal and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND price < 1d EMA34 AND volume spike
            elif short_signal and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price crosses below 1d EMA34
            if (i > 50 and williams_r[i-1] < -20 and williams_r[i] >= -20) or close[i] <= ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price crosses above 1d EMA34
            if (i > 50 and williams_r[i-1] > -80 and williams_r[i] <= -80) or close[i] >= ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals