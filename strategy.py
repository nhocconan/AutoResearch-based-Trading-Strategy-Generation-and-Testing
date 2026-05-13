#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_R1S1_Breakout_4hTrend_DailyVol
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for 1h breakout entries with 4h trend filter and daily volume confirmation. 
This combines intraday precision with higher-timeframe structure to reduce false signals. 
Target: 15-35 trades/year by requiring confluence of 1h breakout, 4h trend alignment, and elevated daily volume.
Works in bull/bear by following 4h trend direction and using volume to confirm institutional participation.
"""

name = "1h_Camarilla_Pivot_R1S1_Breakout_4hTrend_DailyVol"
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
    
    # Calculate 1-hour Camarilla pivot levels from previous day
    # We'll use daily OHLC from previous day to calculate today's Camarilla levels
    # For simplicity in 1h timeframe, we calculate daily pivot once per day and use it for all 1h bars of that day
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day: R1, S1
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are daily close, high, low
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (will use previous day's levels for current day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h trend filter: EMA 50
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = daily_volume > (1.5 * vol_ma_20d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for 4h EMA50
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with 4h uptrend and high volume day
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 with 4h downtrend and high volume day
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below Camarilla R1 or 4h trend turns down
            if close[i] < camarilla_r1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses back above Camarilla S1 or 4h trend turns up
            if close[i] > camarilla_s1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals