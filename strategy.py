#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses actual Camarilla pivot levels from prior 1d session: R3 = H + 1.1*(H-L), S3 = L - 1.1*(H-L)
# Breakout above R3 in 1d uptrend with volume spike = long, breakdown below S3 in 1d downtrend with volume spike = short
# Designed for 12-30 trades/year on 12h to minimize fee drag while capturing strong intraday moves
# Works in both bull and bear markets by following 1d trend direction with confluence filters

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 1d Camarilla levels (R3, S3) - use completed 1d bar only
    # Need high, low, close from prior completed 1d bar
    prior_high = df_1d['high'].shift(1).values  # Prior 1d high
    prior_low = df_1d['low'].shift(1).values    # Prior 1d low
    prior_close = df_1d['close'].shift(1).values # Prior 1d close
    
    # Camarilla R3 and S3 from prior 1d session
    camarilla_r3 = prior_high + 1.1 * (prior_high - prior_low)
    camarilla_s3 = prior_low - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 12h timeframe (wait for prior 1d bar to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA on 12h
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA34
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5 * 20-period EMA
        volume_spike = volume[i] > (1.5 * volume_ema_20[i])
        
        if position == 0:
            # Long: break above Camarilla R3 in 1d uptrend with volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_34_1d_aligned[i] > close[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 in 1d downtrend with volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_34_1d_aligned[i] < close[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Camarilla R3 or loses 1d uptrend
            if (close[i] < camarilla_r3_aligned[i] or 
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Camarilla S3 or loses 1d downtrend
            if (close[i] > camarilla_s3_aligned[i] or 
                ema_34_1d_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals