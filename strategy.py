#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout + 12h Trend + Volume Spike
# Camarilla pivots from 12h timeframe provide intraday support/resistance levels.
# Breakout at R3 (short) or S3 (long) with 12h EMA34 trend filter and volume confirmation (>2x 20-period EMA volume).
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 12h data for Camarilla pivots and EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivots (based on previous 12h bar's high, low, close)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_12h + (h_12h - l_12h) * 1.1 / 4
    camarilla_s3 = c_12h - (h_12h - l_12h) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(c_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above S3 + volume + 12h EMA34 uptrend
            if (close[i] > camarilla_s3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below R3 + volume + 12h EMA34 downtrend
            elif (close[i] < camarilla_r3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR 12h EMA34 turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR 12h EMA34 turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals