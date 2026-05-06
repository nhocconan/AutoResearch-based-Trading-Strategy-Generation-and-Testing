#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 level AND 1d EMA34 > price AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 level AND 1d EMA34 < price AND volume > 2.0 * avg_volume(20)
# Exit when price reaches 1d Camarilla R4/S4 levels (continuation) or returns to R3/S3 (mean reversion)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Camarilla provides institutional support/resistance levels that work in both ranging and trending markets
# 1d EMA34 filter ensures alignment with daily trend, reducing counter-trend trades
# Volume spike confirmation filters weak breakouts and ensures institutional participation

name = "6h_1dCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34 and pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #                S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Using previous day's values, so we shift by 1
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First value will be invalid due to roll, but min_periods will handle it
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_s3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_r4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_s4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Calculate 1d EMA34 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 1d EMA34 > price and volume spike
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > close[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 1d EMA34 < price and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < close[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches Camarilla R4 (continuation) or returns to R3 (mean reversion)
            if close[i] >= camarilla_r4_aligned[i] or close[i] <= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches Camarilla S4 (continuation) or returns to S3 (mean reversion)
            if close[i] <= camarilla_s4_aligned[i] or close[i] >= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals