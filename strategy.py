#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout + 4h Trend + Volume Spike
# Camarilla pivots from 4h timeframe provide intraday support/resistance levels.
# Breakout at R3 (short) or S3 (long) with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period EMA volume).
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year).
# Uses discrete position sizing (0.20) to minimize fee churn and manage drawdown.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead and TypeError
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for Camarilla pivots and EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar's high, low, close)
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_4h + (h_4h - l_4h) * 1.1 / 4
    camarilla_s3 = c_4h - (h_4h - l_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(c_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0 and in_session:
            # Long breakout: price closes above S3 + volume + 4h EMA50 uptrend
            if (close[i] > camarilla_s3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short breakout: price closes below R3 + volume + 4h EMA50 downtrend
            elif (close[i] < camarilla_r3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR 4h EMA50 turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R3 OR 4h EMA50 turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Outside session or flat: remain flat
            signals[i] = 0.0
            position = 0
    
    return signals