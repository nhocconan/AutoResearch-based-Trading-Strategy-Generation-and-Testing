#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure, 1w EMA34 for higher timeframe trend filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d.
# 1w EMA34 provides strong trend filter, reducing whipsaw while capturing major moves in both bull and bear markets.
# Camarilla pivots work well when combined with volume and trend filters, especially in ranging/transitioning markets.

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter from prior completed 1w bar
    close_1w = df_1w['close'].values
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    ema_34_1w = pd.Series(close_1w_shifted).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for prior completed 1d bar (R3, S3, R4, S4)
    # Need high, low, close from prior completed daily bar
    high_1d = pd.Series(high).rolling(window=2, min_periods=2).max().values  # 2-bar window to get prior bar's high
    low_1d = pd.Series(low).rolling(window=2, min_periods=2).min().values    # 2-bar window to get prior bar's low
    close_1d = pd.Series(close).rolling(window=2, min_periods=2).last().values  # 2-bar window to get prior bar's close
    
    # Shift by 1 to use only prior completed bar (avoid look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3
    range_hl = high_1d_shifted - low_1d_shifted
    
    # Camarilla levels: R3, S3, R4, S4
    r3 = close_1d_shifted + range_hl * 1.1 / 4
    s3 = close_1d_shifted - range_hl * 1.1 / 4
    r4 = close_1d_shifted + range_hl * 1.1 / 2
    s4 = close_1d_shifted - range_hl * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1w EMA34 + volume spike
            if close[i] > r3[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 1w EMA34 + volume spike
            elif close[i] < s3[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3-S3 OR price crosses below 1w EMA34
            midpoint = (r3[i] + s3[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3-S3 OR price crosses above 1w EMA34
            midpoint = (r3[i] + s3[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals