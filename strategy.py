# 1d_Weekly_Pivot_Breakout_Trend_Volume_v2
# Hypothesis: Use weekly pivot points (R1/S1) for breakout entries with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above weekly R1 in uptrend with volume spike, short when price breaks below weekly S1 in downtrend with volume spike.
# Exit when price returns to weekly pivot point (PP) or trend changes.
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts.
# Designed for low trade frequency (30-100 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "1d_Weekly_Pivot_Breakout_Trend_Volume_v2"
timeframe = "1d"
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

    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP, R1, S1
    # Standard pivot point formulas:
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r1_1w = (2 * pp_1w) - df_1w['low']
    s1_1w = (2 * pp_1w) - df_1w['high']
    
    # Align weekly pivot levels to 1d timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w.values)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w.values)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w.values)

    # Get daily data for EMA trend filter (200-period for long-term trend)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S1 + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA200)
            if (close[i] <= pp_1w_aligned[i] or close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA200)
            if (close[i] >= pp_1w_aligned[i] or close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals