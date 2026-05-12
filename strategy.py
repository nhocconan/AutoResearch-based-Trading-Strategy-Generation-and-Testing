#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend
Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe act as significant support/resistance on 12h.
Breakouts above R3 or below S3 with daily trend filter (EMA50) capture strong momentum moves.
Volume confirmation filters out false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag.
Works in bull markets via breakout momentum and in bear markets via short breakdowns with trend alignment.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given period.
    R4 = Close + ((High - Low) * 1.500)
    R3 = Close + ((High - Low) * 1.250)
    R2 = Close + ((High - Low) * 1.166)
    R1 = Close + ((High - Low) * 1.083)
    PP = (High + Low + Close) / 3
    S1 = Close - ((High - Low) * 1.083)
    S2 = Close - ((High - Low) * 1.166)
    S3 = Close - ((High - Low) * 1.250)
    S4 = Close - ((High - Low) * 1.500)
    """
    range_hl = high - low
    close_price = close
    r3 = close_price + (range_hl * 1.250)
    s3 = close_price - (range_hl * 1.250)
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily Camarilla levels (R3, S3)
    r3_1d, s3_1d = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period moving average of volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily levels to 12h timeframe (wait for daily close)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        # Get aligned values for current 12h bar
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma20[i]
        vol_current = volume[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(ema50_val) or np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + daily uptrend + volume confirmation
            if (close[i] > r3_val and 
                close[i] > ema50_val and 
                vol_current > vol_ma_val):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + daily downtrend + volume confirmation
            elif (close[i] < s3_val and 
                  close[i] < ema50_val and 
                  vol_current > vol_ma_val):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns down
            if (close[i] < r3_val or close[i] < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns up
            if (close[i] > s3_val or close[i] > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals