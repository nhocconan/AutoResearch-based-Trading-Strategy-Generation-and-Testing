#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Camarilla pivots from 12h provide intraday support/resistance levels.
# Breakout at R3/S3 with 12h trend alignment captures momentum in trending markets.
# Volume spike filters for institutional participation. Designed for 12-30 trades/year on 6h.
# Works in bull markets via R3 breakouts and in bear markets via S3 breakdowns.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r3 = np.zeros(len(df_12h))
    camarilla_s3 = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        # Camarilla calculations using previous 12h bar
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            # Previous 12h bar's range
            prev_high = high_12h[i-1]
            prev_low = low_12h[i-1]
            prev_close = close_12h[i-1]
            range_val = prev_high - prev_low
            
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: 20-period EMA
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above R3 in 12h uptrend with volume spike
            if breakout_up and ema_34_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below S3 in 12h downtrend with volume spike
            elif breakout_down and ema_34_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3-S3 or loses 12h uptrend
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] < midpoint or ema_34_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3-S3 or loses 12h downtrend
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] > midpoint or ema_34_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals