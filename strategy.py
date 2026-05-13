#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike (>2.0x 20-bar avg volume).
# Uses tighter Camarilla levels (R3/S3) for stronger breakout signals, 4h EMA34 for trend alignment,
# and high volume threshold to filter false breakouts. Designed for low trade frequency (<150 total 1h trades over 4 years)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets. Session filter (08-20 UTC) added to reduce noise.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    # R3 = close + 1.1*(high-low)*1.125/4
    # S3 = close - 1.1*(high-low)*1.125/4
    prior_4h_high = df_4h['high'].values
    prior_4h_low = df_4h['low'].values
    prior_4h_close = df_4h['close'].values
    
    camarilla_r3 = prior_4h_close + 1.1 * (prior_4h_high - prior_4h_low) * 1.125 / 4
    camarilla_s3 = prior_4h_close - 1.1 * (prior_4h_high - prior_4h_low) * 1.125 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 4h EMA34, volume spike (>2.0x avg)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.20  # Conservative size to minimize fee drag
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 4h EMA34, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.20  # Conservative size to minimize fee drag
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla R3 or volume drops
            if (low[i] < camarilla_r3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # Maintain position
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla S3 or volume drops
            if (high[i] > camarilla_s3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # Maintain position
    
    return signals