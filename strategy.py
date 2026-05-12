# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Breakouts at 4h Camarilla R1/S1 levels with volume confirmation and 1d trend filter on 1h timeframe.
# Targets 15-37 trades/year to stay within fee limits. Uses 4h for direction and 1d for trend filter.
# Long: Close > 4h R1 + volume > 1.5x SMA20 + price > 1d EMA50
# Short: Close < 4h S1 + volume > 1.5x SMA20 + price < 1d EMA50
# Exit: Close crosses opposite 4h Camarilla level (S1 for long, R1 for short)

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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

    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate Camarilla levels from previous 4h close
    camarilla_range = high_4h - low_4h
    r1 = close_4h + camarilla_range * 1.1 / 12
    s1 = close_4h - camarilla_range * 1.1 / 12

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 1h bar
        r1_aligned = align_htf_to_ltf(prices, df_4h, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_4h, s1)[i]
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above 4h R1 + volume spike (1.5x) + daily uptrend
            if (close[i] > r1_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.20
                position = 1
            # SHORT: Price closes below 4h S1 + volume spike (1.5x) + daily downtrend
            elif (close[i] < s1_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h S1
            if close[i] < s1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h R1
            if close[i] > r1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals