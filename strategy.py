#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Camarilla levels calculated from previous 1d OHLC: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
# Long when: price breaks above R3 AND 1w close > 1w EMA34 AND volume > 2x 20-period MA
# Short when: price breaks below S3 AND 1w close < 1w EMA34 AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla H3/L3 levels (mean reversion toward daily equilibrium)
# Uses Camarilla structure for institutional levels, 1w EMA for major trend alignment, volume for conviction
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for Camarilla levels (daily pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least 2 days for previous day calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range  # Resistance level 3
    s3 = prev_close - 1.1 * camarilla_range  # Support level 3
    h3 = prev_close + 1.0625 * camarilla_range  # Resistance level 3 (quarter)
    l3 = prev_close - 1.0625 * camarilla_range  # Support level 3 (quarter)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 AND above 1w EMA34 AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 AND below 1w EMA34 AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to H3 level (mean reversion toward daily equilibrium)
            if close[i] <= h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to L3 level (mean reversion toward daily equilibrium)
            if close[i] >= l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals