#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance levels. 
# Long when price breaks above R3 with volume spike and daily uptrend, short when price breaks below S3 with volume spike and daily downtrend.
# Uses daily Camarilla levels calculated from prior day's OHLC, volume > 1.5x 20-period average for confirmation,
# and daily EMA50 for trend filter. Designed for 4h timeframe to limit trades and avoid overtrading.
# Works in bull markets via breakouts in uptrends and in bear markets via breakdowns in downtrends.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from prior day's OHLC
    # Camarilla formulas: 
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.1250)
    # R2 = Close + ((High - Low) * 1.0000)
    # R1 = Close + ((High - Low) * 0.5000)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 0.5000)
    # S2 = Close - ((High - Low) * 1.0000)
    # S3 = Close - ((High - Low) * 1.1250)
    # S4 = Close - ((High - Low) * 1.5000)
    
    # We need prior day's data to calculate today's levels (no look-ahead)
    # Shift the OHLC data by 1 to use previous day's values
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels using prior day's data
    R3 = prev_close + ((prev_high - prev_low) * 1.1250)
    S3 = prev_close - ((prev_high - prev_low) * 1.1250)
    
    # Handle first day where we don't have prior data
    R3[0] = np.nan
    S3[0] = np.nan

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R3 with volume spike and daily uptrend
            if close[i] > R3_aligned[i] and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and daily downtrend
            elif close[i] < S3_aligned[i] and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < S3_aligned[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > R3_aligned[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals