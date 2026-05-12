#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from 1-day timeframe provide key support/resistance.
# Breakout above R3 or below S3 with volume confirmation and daily trend alignment
# leads to strong continuation moves. Works in bull markets via breakouts in uptrends
# and in bear markets via breakdowns in downtrends. Uses daily Camarilla levels
# calculated from prior day's OHLC, aligned to 6h chart, with volume > 1.5x 20-period
# average for confirmation. Designed for 6h timeframe to balance trade frequency
# and avoid excessive fee churn.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily Camarilla levels from prior day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of prior day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's data (avoid look-ahead)
    close_1d_prior = np.roll(close_1d, 1)
    high_1d_prior = np.roll(high_1d, 1)
    low_1d_prior = np.roll(low_1d, 1)
    # Set first value to NaN since no prior day exists
    close_1d_prior[0] = np.nan
    high_1d_prior[0] = np.nan
    low_1d_prior[0] = np.nan
    
    camarilla_r3 = close_1d_prior + (high_1d_prior - low_1d_prior) * 1.1 / 2
    camarilla_s3 = close_1d_prior - (high_1d_prior - low_1d_prior) * 1.1 / 2
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    price_above_daily_ema = close_1d > ema_34_1d
    price_below_daily_ema = close_1d < ema_34_1d

    # Align all daily data to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    price_above_daily_ema_aligned = align_htf_to_ltf(prices, df_1d, price_above_daily_ema.astype(float))
    price_below_daily_ema_aligned = align_htf_to_ltf(prices, df_1d, price_below_daily_ema.astype(float))

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(price_above_daily_ema_aligned[i]) or np.isnan(price_below_daily_ema_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]

        if position == 0:
            # LONG: Breakout above R3 with volume and daily uptrend
            if breakout_up and volume_ok[i] and price_above_daily_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with volume and daily downtrend
            elif breakout_down and volume_ok[i] and price_below_daily_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns down
            if close[i] < camarilla_r3_aligned[i] or not price_above_daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns up
            if close[i] > camarilla_s3_aligned[i] or not price_below_daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals