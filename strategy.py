# 4h Camarilla R1/S1 Breakout + Volume Spike + Daily Trend Filter
# Hypothesis: Using tighter R1/S1 levels from daily data with volume confirmation and daily trend filter
# provides higher probability entries with lower frequency than R3/S3, reducing whipsaw in ranging markets.
# The R1/S1 levels are closer to price, capturing earlier momentum while the volume spike and trend filter
# ensure we only trade strong moves. Designed for 15-25 trades/year to minimize fee decay.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) when aligned with trend.

name = "4h_Camarilla_R1S1_Breakout_DailyTrend_Volume_v2"
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
    
    # === Daily Data for Camarilla Levels and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily data (previous day's HLC)
    daily_high_1d = df_1d['high'].values
    daily_low_1d = df_1d['low'].values
    daily_close_1d = df_1d['close'].values
    
    rng = daily_high_1d - daily_low_1d
    R1 = daily_close_1d + rng * 1.1 / 12  # R1 = C + (H-L)*1.1/12
    S1 = daily_close_1d - rng * 1.1 / 12  # S1 = C - (H-L)*1.1/12
    
    # Shift to get previous day's levels (today's R1/S1 based on yesterday's HLC)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1_prev)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1_prev)
    
    # === Daily EMA34 for trend filter ===
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > R1_4h[i] and 
                vol_spike[i] and
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < S1_4h[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (reversal)
            if close[i] < S1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (reversal)
            if close[i] > R1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals