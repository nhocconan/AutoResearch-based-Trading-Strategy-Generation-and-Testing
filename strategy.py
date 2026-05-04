#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability support/resistance in ranging markets.
# Breakouts above R3 or below S3 with volume confirmation and 1d EMA34 trend alignment capture strong moves.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Uses discrete position sizing (0.30) to balance return and drawdown.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) from previous day
    # Camarilla: R4 = close + ((high - low) * 1.5/2), R3 = close + ((high - low) * 1.25/2)
    #          S3 = close - ((high - low) * 1.25/2), S4 = close - ((high - low) * 1.5/2)
    # We use R3 and S3 as key breakout levels
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_range = prev_high - prev_low
    
    camarilla_r3 = prev_close + (prev_range * 1.25 / 2)
    camarilla_s3 = prev_close - (prev_range * 1.25 / 2)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above R3 with volume and EMA34 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short breakout: price closes below S3 with volume and EMA34 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retouches EMA34 OR volume drops below average
            if (close[i] <= ema_34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price retouches EMA34 OR volume drops below average
            if (close[i] >= ema_34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals