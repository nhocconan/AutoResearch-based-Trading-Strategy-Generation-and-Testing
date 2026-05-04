#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Weekly Camarilla R4 level with 1d EMA34 uptrend and volume > 2.5x 20-period volume EMA
# Short when price breaks below Weekly Camarilla S4 level with 1d EMA34 downtrend and volume > 2.5x 20-period volume EMA
# Weekly Camarilla provides stronger institutional levels than daily, reducing false breakouts.
# 1d EMA34 trend filter avoids counter-trend trades in choppy markets.
# Volume spike confirmation (2.5x) ensures institutional participation.
# Designed for 6h timeframe to capture multi-day trends with lower trade frequency (target: 12-30 trades/year).
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "6h_WeeklyCamarilla_R4S4_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Weekly Camarilla levels using previous week's OHLC
    # Camarilla: R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    prev_weekly_high = np.roll(df_1w['high'].values, 1)
    prev_weekly_low = np.roll(df_1w['low'].values, 1)
    prev_weekly_close = np.roll(df_1w['close'].values, 1)
    prev_weekly_high[0] = df_1w['high'].values[0]
    prev_weekly_low[0] = df_1w['low'].values[0]
    prev_weekly_close[0] = df_1w['close'].values[0]
    
    camarilla_r4 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1
    camarilla_s4 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1
    
    # Align Weekly Camarilla levels to 6h timeframe (waits for weekly bar close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.5)  # Volume at least 2.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Weekly Camarilla R4 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Weekly Camarilla S4 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Weekly Camarilla S4 OR 1d trend turns down
            if (close[i] < camarilla_s4_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Weekly Camarilla R4 OR 1d trend turns up
            if (close[i] > camarilla_r4_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals