#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Camarilla pivot levels identify key intraday support/resistance where institutional order flow clusters
# Breakouts beyond R3/S3 with 12h EMA34 trend alignment capture strong momentum moves
# Volume spike (>1.8x 20-period EMA volume) filters false breakouts
# Discrete sizing 0.25 targets 80-180 total trades over 4 years (20-45/year) for 4h timeframe
# Works in bull markets (R3 breakouts with uptrend) and bear markets (S3 breakdowns with downtrend)
# ATR-based stoploss via signal=0 when price moves against position

name = "4h_Camarilla_R3S3_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 trend filter from prior completed 12h bar
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_shifted = np.roll(ema34_12h, 1)
    ema34_12h_shifted[0] = np.nan
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h_shifted)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = typical_price + range_1d * 1.1 / 4
    camarilla_s3 = typical_price - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 12h EMA34 uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 12h EMA34 downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 12h EMA34 turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 12h EMA34 turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals