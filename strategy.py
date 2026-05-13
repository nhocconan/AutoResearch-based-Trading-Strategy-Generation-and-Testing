#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with 1w EMA50 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 level with 1w EMA50 downtrend and volume > 2.0x average.
# Exit when price reverses and crosses the Camarilla pivot point (PP).
# Uses discrete position sizing 0.30. Target: 50-150 total trades over 4 years on 12h timeframe.
# The 1w EMA50 filter ensures we only trade in the dominant weekly trend, reducing whipsaws.
# Volume confirmation validates breakout strength. Camarilla pivot exit provides clear, objective stop.

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (high + low + close) / 3
    # R3 = PP + (high - low) * 1.1 / 2
    # S3 = PP - (high - low) * 1.1 / 2
    # We use the previous bar's values to avoid look-ahead
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3
    r3 = pp + (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    s3 = pp - (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    
    # Set first value to NaN since we don't have previous bar
    pp[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w data
    if len(close_1w) < 50:
        ema_50_1w = np.full(len(close_1w), np.nan)
    else:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume average
        # Skip if any required data is NaN
        if (np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with 1w EMA50 uptrend and volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_50_1w_aligned[i] and  # Uptrend: price above EMA50
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 with 1w EMA50 downtrend and volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # Downtrend: price below EMA50
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below PP (reversal signal)
            if close[i] < pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above PP (reversal signal)
            if close[i] > pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals