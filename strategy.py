#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivots identify key support/resistance levels. Breakouts above R3 or below S3
# with volume confirmation and 1d EMA34 trend alignment capture strong momentum moves.
# Designed for 6h timeframe to achieve 12-37 trades/year (50-150 total over 4 years)
# Works in both bull and bear markets by trading breakouts in the direction of the 1d trend.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivots (using previous day's OHLC)
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_r3 = typical_price + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = typical_price - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after sufficient warmup for 1d EMA34
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5 * 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 1d uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_34_1d_aligned[i] > close[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and 1d downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_34_1d_aligned[i] < close[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 1d uptrend
            if (close[i] < camarilla_r3_aligned[i] or 
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 1d downtrend
            if (close[i] > camarilla_s3_aligned[i] or 
                ema_34_1d_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals