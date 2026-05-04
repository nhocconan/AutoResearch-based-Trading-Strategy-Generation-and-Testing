#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points. Breakout above R3 or below S3
# with 1d EMA34 trend alignment and volume spike (>2x 20 EMA) captures strong momentum moves.
# Discrete sizing 0.30 balances risk and return. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull/bear: uses 1d trend filter to avoid counter-trend breakouts.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (for 12h chart)
    # Camarilla: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We'll use daily high/low/close from previous completed 1d bar
    prev_close_1d = df_1d['close'].shift(1).values  # previous day close
    prev_high_1d = df_1d['high'].shift(1).values    # previous day high
    prev_low_1d = df_1d['low'].shift(1).values      # previous day low
    
    # Calculate Camarilla levels for previous day
    prev_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + 1.1 * prev_range
    camarilla_s3 = prev_close_1d - 1.1 * prev_range
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: break above R3 + uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short conditions: break below S3 + downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range OR trend changes OR volume drops
            camarilla_h = camarilla_r3_aligned[i]  # R3 as upper bound
            camarilla_l = camarilla_s3_aligned[i]  # S3 as lower bound
            if (close[i] < camarilla_h and close[i] > camarilla_l) or \
               close[i] < ema34_aligned[i] or \
               volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR trend changes OR volume drops
            camarilla_h = camarilla_r3_aligned[i]  # R3 as upper bound
            camarilla_l = camarilla_s3_aligned[i]  # S3 as lower bound
            if (close[i] < camarilla_h and close[i] > camarilla_l) or \
               close[i] > ema34_aligned[i] or \
               volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals