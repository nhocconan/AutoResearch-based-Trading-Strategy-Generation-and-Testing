#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 Breakout + 1w Trend + Volume Spike
# Camarilla pivots from 1d timeframe provide key support/resistance levels.
# Breakout at R3 (short) or S3 (long) with 1w EMA34 trend filter and volume confirmation (>2x 20-period EMA volume).
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for Camarilla pivots and EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivots (based on previous 1w bar's high, low, close)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1w + (h_1w - l_1w) * 1.1 / 4
    camarilla_s3 = c_1w - (h_1w - l_1w) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above S3 + volume + 1w EMA34 uptrend
            if (close[i] > camarilla_s3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below R3 + volume + 1w EMA34 downtrend
            elif (close[i] < camarilla_r3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR 1w EMA34 turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR 1w EMA34 turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals