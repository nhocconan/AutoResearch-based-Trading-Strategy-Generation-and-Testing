# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Strategy: Use Camarilla R3/S3 levels from 1d as breakout levels, with 1d trend filter (EMA50) and volume confirmation.
# In trending markets, price breaks R3 (bullish) or S3 (bearish) with volume, filtered by 1d EMA50 direction.
# Ranges: price stays between S3 and R3, mean-reverting at boundaries.
# Targets 15-25 trades/year by requiring breakout + trend + volume confluence.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla levels and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels for previous day (using OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), etc.
    # We use R3 and S3 for breakouts
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = daily_close + ((daily_high - daily_low) * 1.25 / 2)
    camarilla_s3 = daily_close - ((daily_high - daily_low) * 1.25 / 2)
    
    # Trend filter: 50 EMA on daily close
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)

    # Volume confirmation: current volume > 1.5x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: price above/below EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: breakout above R3 + price above EMA50 + volume
            if breakout_up and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below S3 + price below EMA50 + volume
            elif breakout_down and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below S3 (mean reversion) OR trend reversal
            if close[i] < camarilla_s3_aligned[i] or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R3 (mean reversion) OR trend reversal
            if close[i] > camarilla_r3_aligned[i] or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals