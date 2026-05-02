#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) for structure, 1d EMA34 for trend alignment to avoid counter-trend entries
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 20-50 trades/year (75-200 total over 4 years) for 4h timeframe
# Works in bull markets via upper Camarilla breakout continuation and in bear markets via lower Camarilla breakdown continuation

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    price_range = high - low
    # Camarilla R3 = close + (range * 1.1 / 4)
    camarilla_r3 = close + (price_range * 1.1 / 4.0)
    # Camarilla S3 = close - (range * 1.1 / 4)
    camarilla_s3 = close - (price_range * 1.1 / 4.0)
    # Shift by 1 to use previous day's levels
    camarilla_r3_shifted = camarilla_r3.shift(1).values
    camarilla_s3_shifted = camarilla_s3.shift(1).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_shifted[i]) or 
            np.isnan(camarilla_s3_shifted[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 1d close > EMA34 + volume spike
            if (close[i] > camarilla_r3_shifted[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 1d close < EMA34 + volume spike
            elif (close[i] < camarilla_s3_shifted[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 or 1d trend breaks
            if close[i] < camarilla_s3_shifted[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 or 1d trend breaks
            if close[i] > camarilla_r3_shifted[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals