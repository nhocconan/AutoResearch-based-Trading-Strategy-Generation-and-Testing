# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot breakout at R3/S3 levels from daily data with 1-day trend filter and volume spike (>1.5x 20-period avg) captures high-probability moves in both bull and bear markets.
# Long when price breaks above R3 + price > 1d EMA34 + volume spike. Short when price breaks below S3 + price < 1d EMA34 + volume spike.
# Exit when price reverses to touch opposite Camarilla level (R2/S2) or trend changes.
# Designed for low trade frequency (target 15-35/year) to minimize fee decay while capturing strong directional moves.
# Uses daily Camarilla levels which adapt to volatility, providing dynamic support/resistance.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+C)/3 (typical price), but for pivot we use previous day's close as base
    # Standard Camarilla uses previous day's close as the base calculation point
    
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First value fallback
    
    # Calculate Camarilla levels using previous day's data
    rang = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + (rang * 1.1 / 4)
    s3 = prev_close_1d - (rang * 1.1 / 4)
    r2 = prev_close_1d + (rang * 1.1 / 6)
    s2 = prev_close_1d - (rang * 1.1 / 6)
    
    # Align Camarilla levels to 6h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)

    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + 1-day uptrend + volume spike
            if close[i] > r3_aligned[i-1] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1-day downtrend + volume spike
            elif close[i] < s3_aligned[i-1] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or goes below R2 OR trend turns down
            if close[i] <= r2_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or goes above S2 OR trend turns up
            if close[i] >= s2_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals