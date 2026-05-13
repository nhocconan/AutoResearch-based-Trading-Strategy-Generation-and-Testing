# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Use daily (1d) price action to filter 12h Camarilla R3/S3 breakouts. Long when price breaks above R3 in a daily uptrend with volume spike, short when price breaks below S3 in a daily downtrend with volume spike. Exit when price returns to 12h pivot point (PP). Designed for low trade frequency to avoid fee drag, with clear trend filter to work in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels: R3, S3, and PP (pivot point)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    pp_12h = typical_price.values
    hl_range = df_12h['high'] - df_12h['low']
    r3_12h = df_12h['close'].values + hl_range.values * 1.1 / 2
    s3_12h = df_12h['close'].values - hl_range.values * 1.1 / 2
    
    # Align 12h Camarilla levels to 12h timeframe (no change needed as already aligned)
    r3_12h_aligned = r3_12h
    s3_12h_aligned = s3_12h
    pp_12h_aligned = pp_12h

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (using daily close)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average (on 12h timeframe)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(pp_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above 1d EMA50 (uptrend) + volume spike
            if (close[i] > r3_12h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below 1d EMA50 (downtrend) + volume spike
            elif (close[i] < s3_12h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_12h_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_12h_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals