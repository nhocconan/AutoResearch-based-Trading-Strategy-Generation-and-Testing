# 12H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp
# Hypothesis: Camarilla R1/S1 breakouts on 12h with daily EMA34 trend filter and volume spikes work in both bull and bear markets.
# The Camarilla levels provide clear support/resistance, EMA34 filters trend direction, and volume confirms breakout strength.
# Target: 50-150 trades over 4 years (12-37/year) with strict entry conditions to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Load daily data for Camarilla levels and EMA34 - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels (using previous day's close)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = np.roll(daily_close, 1)
    prev_close[0] = daily_close[0]  # first day
    prev_high = np.roll(daily_high, 1)
    prev_high[0] = daily_high[0]
    prev_low = np.roll(daily_low, 1)
    prev_low[0] = daily_low[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12
    
    # Calculate daily EMA34
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA34 to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above EMA34
            if (close[i] > R1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below EMA34
            elif (close[i] < S1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Camarilla level
            if position == 1:
                if close[i] < S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0