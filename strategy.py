#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as key support/resistance.
# Enter long when price breaks above R1 with volume confirmation and 12h uptrend (price > EMA50).
# Enter short when price breaks below S1 with volume confirmation and 12h downtrend (price < EMA50).
# Exit when price returns to the mean (close of previous day) or reverses.
# Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 20-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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

    # Calculate daily Camarilla pivot levels (based on previous day)
    # We'll use daily data to calculate R1, S1, and the mean (close)
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Actually, standard Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # But we'll use the more common definition: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # However, looking at successful strategies, they often use: R1 = close + (high - low) * 1.1/6, S1 = close - (high - low) * 1.1/6
    # Let's use the standard Camarilla definition for R1 and S1:
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # But to match the successful patterns, let's use the multiplier that worked: 1.1/6 for wider bands
    # Actually, from the database, successful strategies use the standard Camarilla calculation
    # Let's stick to: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # But we saw that the best performers used the standard definition
    
    # Calculate the Camarilla levels for each day
    camarilla_multiplier = 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * camarilla_multiplier
    s1 = prev_close - (prev_high - prev_low) * camarilla_multiplier
    # The mean (close) is used for exit
    daily_mean = prev_close
    
    # Align the daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    daily_mean_aligned = align_htf_to_ltf(prices, df_1d, daily_mean)

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(daily_mean_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and 12h uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_avg_30[i] * 1.5 and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and 12h downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_avg_30[i] * 1.5 and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily mean or closes below S1 (reversal)
            if (close[i] <= daily_mean_aligned[i] or close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily mean or closes above R1 (reversal)
            if (close[i] >= daily_mean_aligned[i] or close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals