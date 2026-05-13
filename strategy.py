#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Squeeze
# Hypothesis: Use Camarilla pivot points from daily data for breakout entries, filtered by daily trend and volume squeeze conditions.
# Long when price breaks above R1 during daily uptrend with volume squeeze release.
# Short when price breaks below S1 during daily downtrend with volume squeeze release.
# Exit on opposite Camarilla level touch or trend reversal.
# Designed for low trade frequency (<40 total/year) to minimize fee drag and work in both bull and bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Squeeze"
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

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily OHLC
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666), R1 = C + ((H-L) * 1.0833)
    # PP = (H+L+C)/3, S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666), S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    R1 = C + (H - L) * 1.0833
    S1 = C - (H - L) * 1.0833
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(C).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume squeeze: Bollinger Bands width < 20th percentile indicates low volatility
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_std_20 = vol_series.rolling(window=20, min_periods=20).std().values
    vol_upper = vol_ma_20 + 2 * vol_std_20
    vol_lower = vol_ma_20 - 2 * vol_std_20
    bb_width = (vol_upper - vol_lower) / vol_ma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).rank(pct=True).values
    volatility_squeeze = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # Align all daily data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volatility_squeeze_aligned = align_htf_to_ltf(prices, df_1d, volatility_squeeze.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volatility_squeeze_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 during daily uptrend with volatility squeeze release
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                if close[i] > ema_34_1d_aligned[i] and volatility_squeeze_aligned[i] > 0.3:  # Squeeze releasing
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 during daily downtrend with volatility squeeze release
            elif close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                if close[i] < ema_34_1d_aligned[i] and volatility_squeeze_aligned[i] > 0.3:  # Squeeze releasing
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Touch S1 or trend turns down
            if close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:  # Trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Touch R1 or trend turns up
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:  # Trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals