#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot breakout with volume confirmation and chop regime filter
# Long when price breaks above 12h Camarilla R3 level AND volume > 1.3 * avg_volume(20) AND chop > 61.8 (range)
# Short when price breaks below 12h Camarilla S3 level AND volume > 1.3 * avg_volume(20) AND chop > 61.8 (range)
# Exit when price crosses 12h EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Camarilla provides clear structure with proven breakout edge in ranging markets
# Volume confirmation filters weak breakouts (reduces false signals)
# Chop regime filter ensures we only trade in ranging conditions where Camarilla works best

name = "4h_12hCamarillaR3S3_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels based on previous 12h bar
    # Camarilla: R3 = Close + 1.125 * (High - Low), S3 = Close - 1.125 * (High - Low)
    camarilla_r3_12h = close_12h + 1.125 * (high_12h - low_12h)
    camarilla_s3_12h = close_12h - 1.125 * (high_12h - low_12h)
    
    # Calculate 12h EMA50 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Chopiness Index (14) on 4h data for regime filter
    # Chop = 100 * log10(sum(ATR(1) over 14) / log10(highest_high - lowest_low over 14)) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First TR is just high-low
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop = 100 * np.log10(sum_atr1_14 / chop_denom) / np.log10(14)
    
    # Align 12h Camarilla levels and EMA to 4h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 with volume confirmation AND chop > 61.8 (range)
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                volume_confirm[i] and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 with volume confirmation AND chop > 61.8 (range)
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  volume_confirm[i] and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal signal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal signal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals