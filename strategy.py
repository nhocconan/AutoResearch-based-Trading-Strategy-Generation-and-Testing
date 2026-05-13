# 4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_Volume
# Hypothesis: Use daily Camarilla R3/S3 levels for breakout entries with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above daily R3 in uptrend with volume spike, short when price breaks below daily S3 in downtrend with volume spike.
# Exit when price returns to daily pivot point or trend changes.
# Daily Camarilla levels provide strong support/resistance; EMA50 filters trend direction; volume confirms breakout strength.
# Designed for low-moderate trade frequency (20-60 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # Camarilla formulas based on previous day's OHLC
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    r3_1d = close_1d + ((high_1d - low_1d) * 1.2500)
    s3_1d = close_1d - ((high_1d - low_1d) * 1.2500)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align daily Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)

    # Get daily data for EMA trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above 1d EMA50 (uptrend) + volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below 1d EMA50 (downtrend) + volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals