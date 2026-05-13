#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_1dVol
# Hypothesis: Price breaking out of Camarilla R3/S3 levels with 4h EMA50 trend filter and 1d volume spike captures strong momentum moves in both bull and bear markets.
# Uses Camarilla pivot points for intraday support/resistance levels. The R3/S3 levels act as significant breakout points.
# Entry: Long when close > R3 + price > 4h EMA50 + 1d volume > 1.5x 20-day average; Short when close < S3 + price < 4h EMA50 + 1d volume > 1.5x 20-day average.
# Exit: Mean reversion to the Camarilla pivot point (center) to avoid overstaying in extended moves.
# Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume periods.
# Target: 15-35 trades/year on 1h to stay within optimal range while capturing significant moves.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVol"
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

    # Calculate Camarilla pivot points for the previous day
    # Typical price = (high + low + close) / 3
    # But for pivot calculation, we use the previous day's data
    # We'll calculate daily pivots using 1d data
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We need to calculate these for each day and then align to 1h
    # But we only need R3 and S3 for breakout
    
    # Calculate daily high, low, close
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.2500
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.2500
    camarilla_pp = (daily_high + daily_low + daily_close) / 3.0
    
    # Align to 1h timeframe (wait for daily bar to close)
    camarilla_r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_1h = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume filter
    # Calculate 20-day average volume on 1d
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1h = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any required value is NaN or not in session
        if (np.isnan(camarilla_r3_1h[i]) or np.isnan(camarilla_s3_1h[i]) or 
            np.isnan(ema50_4h_1h[i]) or np.isnan(vol_avg_20_1h[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + price > 4h EMA50 + volume spike
            if (close[i] > camarilla_r3_1h[i] and 
                close[i] > ema50_4h_1h[i] and
                volume[i] > vol_avg_20_1h[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Close below S3 + price < 4h EMA50 + volume spike
            elif (close[i] < camarilla_s3_1h[i] and 
                  close[i] < ema50_4h_1h[i] and
                  volume[i] > vol_avg_20_1h[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to Camarilla pivot point
            if close[i] < camarilla_pp_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Mean reversion to Camarilla pivot point
            if close[i] > camarilla_pp_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals