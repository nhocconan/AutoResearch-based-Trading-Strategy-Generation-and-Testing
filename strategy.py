#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points. Trading breaks of R3/S3
# with daily trend alignment (price vs 1d EMA34) and 1d volume spike filters for institutional participation.
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
# Works in both bull and bear markets by aligning with higher timeframe trend and requiring volume confirmation.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation, EMA and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate Camarilla levels for previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), S3 = close - 1.25*(high-low)
    camarilla_r3 = df_1d['close'].values + 1.25 * (df_1d['high'].values - df_1d['low'].values)
    camarilla_s3 = df_1d['close'].values - 1.25 * (df_1d['high'].values - df_1d['low'].values)
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R3 in uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 in downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close returns below Camarilla R3 (failed breakout) or trend reversal
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close returns above Camarilla S3 (failed breakdown) or trend reversal
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals