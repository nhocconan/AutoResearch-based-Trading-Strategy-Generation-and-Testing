#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with daily volume confirmation and weekly trend filter
# Long when price breaks above 12h Camarilla resistance level with volume spike and weekly bullish trend
# Short when price breaks below 12h Camarilla support level with volume spike and weekly bearish trend
# Exit when price crosses the Camarilla pivot point
# Uses weekly EMA trend filter to avoid counter-trend trades in bear markets
# Target: 15-35 trades per symbol over 4 years (~4-9/year) to minimize fee drag on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and daily data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels calculation (based on previous period's range)
    # Resistance levels
    camarilla_r4 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 2
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 4
    camarilla_r2 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 6
    camarilla_r1 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 12
    # Pivot point
    camarilla_pivot = (high_12h + low_12h + close_12h) / 3
    # Support levels
    camarilla_s1 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 12
    camarilla_s2 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 6
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 4
    camarilla_s4 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 2
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for volume MA)
    start = 40  # conservative start
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_daily_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above Camarilla R1 with volume spike and weekly bullish trend
            if (price > camarilla_r1_aligned[i] and 
                vol_12h_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla S1 with volume spike and weekly bearish trend
            elif (price < camarilla_s1_aligned[i] and 
                  vol_12h_current > 1.8 * vol_ma_daily_aligned[i] and  # Volume spike
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0