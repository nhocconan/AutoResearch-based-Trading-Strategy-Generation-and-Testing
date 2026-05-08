#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Camarilla levels (S1, S2, R1, R2) provide precise support/resistance in both trending and ranging markets.
# Long when price breaks above R1 in 1d uptrend with volume confirmation.
# Short when price breaks below S1 in 1d downtrend with volume confirmation.
# Uses 1d EMA(34) trend filter to ensure alignment with daily momentum.
# Designed for moderate trade frequency (15-35/year) to balance opportunity with cost efficiency.

name = "1h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels from previous 4h bar
    camarilla_r1 = np.zeros_like(close_4h)  # R1 level
    camarilla_s1 = np.zeros_like(close_4h)  # S1 level
    
    for i in range(1, len(close_4h)):
        # Previous 4h bar's high, low, close
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        
        # Camarilla calculations
        camarilla_r1[i] = pc + (ph - pl) * 1.1 / 12  # R1
        camarilla_s1[i] = pc - (ph - pl) * 1.1 / 12  # S1
    
    # First bar has no previous data
    camarilla_r1[0] = camarilla_s1[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising daily EMA
    daily_trend_up = np.concatenate([[False], daily_trend_up])  # Align with daily index
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 24-period EMA
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(daily_trend_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R1 in daily uptrend with volume and session
            if (daily_trend_aligned[i] > 0.5 and  # Daily uptrend
                close[i] > camarilla_r1_aligned[i] and  # Break above R1
                vol_confirm[i] and
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short setup: break below S1 in daily downtrend with volume and session
            elif (daily_trend_aligned[i] <= 0.5 and  # Daily downtrend
                  close[i] < camarilla_s1_aligned[i] and  # Break below S1
                  vol_confirm[i] and
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break below S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or daily_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or daily_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals