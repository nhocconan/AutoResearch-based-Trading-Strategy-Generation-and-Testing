#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 12h timeframe, price breaking above Camarilla R3 or below S3 with 1d EMA34 trend filter and volume confirmation captures strong momentum moves.
# Works in bull markets (breakouts continuation) and bear markets (mean reversion fails, breakouts still work) by requiring volume and trend alignment.
# Targets 15-25 trades/year by requiring confluence of price level breakout, trend filter, and volume spike.

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We only need R3 and S3 for breakout signals
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar (based on previous day)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 2x average of last 4 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check for breakout above R3 or below S3
        breakout_above = close[i] > camarilla_r3_aligned[i]
        breakout_below = close[i] < camarilla_s3_aligned[i]
        
        # Check trend alignment from 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: breakout above R3 with volume and uptrend
            if breakout_above and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below S3 with volume and downtrend
            elif breakout_below and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below R3 or below EMA34
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above S3 or above EMA34
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals