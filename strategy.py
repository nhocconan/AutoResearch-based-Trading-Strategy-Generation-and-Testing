#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: On 1h timeframe, use 4h EMA50 for trend direction and 1d volume spike for confirmation.
Enter long when price breaks above Camarilla R1 level (from prior 4h) with 4h uptrend and 1d volume > 2x 20-period average.
Enter short when price breaks below Camarilla S1 level with 4h downtrend and volume spike.
Use 1d Bollinger Band width < 60th percentile to avoid choppy markets.
Targets 15-37 trades/year (60-150 total over 4 years) with low turnover.
Works in bull via momentum breaks and bear via mean-reversion at extremes with trend filter.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend and Camarilla levels (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate 1d Bollinger Band width (20, 2) for chop filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    bb_width_rank = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1d, bb_width_rank)

    # Calculate 4h Camarilla levels from previous 4h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    camarilla_mult = 1.1 / 12
    r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * camarilla_mult
    s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * camarilla_mult
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get aligned values for current 1h bar
        ema50 = ema50_4h_aligned[i]
        vol_avg = vol_avg_20_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r1_level = r1_4h_aligned[i]
        s1_level = s1_4h_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(vol_avg) or 
            np.isnan(bb_rank) or np.isnan(r1_level) or 
            np.isnan(s1_level)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Chop filter: only trade when BB width is in lower 60% (avoid extreme chop)
        if bb_rank > 0.6:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 4h uptrend + volume spike
            if (close[i] > r1_level and 
                close[i] > ema50 and 
                volume[i] > vol_avg * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h downtrend + volume spike
            elif (close[i] < s1_level and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 4h trend turns down
            if (close[i] < s1_level or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 4h trend turns up
            if (close[i] > r1_level or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals