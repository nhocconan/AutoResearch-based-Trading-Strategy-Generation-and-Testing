#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND 12h EMA50 rising AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 level AND 12h EMA50 falling AND volume > 1.5 * avg_volume(20)
# Exit when price touches 12h Camarilla pivot point (PP) or opposite S1/R1 level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Camarilla provides strong institutional support/resistance levels with less noise than 1d
# 12h EMA50 trend filter ensures alignment with intermediate-term trend, reducing counter-trend trades
# Moderate volume confirmation (1.5x) filters weak breakouts while allowing sufficient trades
# Works in bull (trend continuation breakouts above R3) and bear (trend continuation breakdowns below S3)

name = "4h_12hCamarilla_R3S3_Breakout_12hEMA50Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla pivots and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on previous bar's OHLC)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    pp_12h = typical_price_12h
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Calculate 12h EMA50 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 values to 4h timeframe (wait for completed 12h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA50 slope for trend direction (rising/falling)
    ema_50_slope = np.zeros_like(ema_50_aligned)
    ema_50_slope[1:] = ema_50_aligned[1:] - ema_50_aligned[:-1]
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_50_slope[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 with 12h EMA50 rising and volume confirmation
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                ema_50_slope[i] > 0 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 with 12h EMA50 falling and volume confirmation
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  ema_50_slope[i] < 0 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 12h Camarilla pivot point (PP) or S1 level (profit take or reversal)
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 12h Camarilla pivot point (PP) or R1 level (profit take or reversal)
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals