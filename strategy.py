#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above weekly Camarilla R4 AND 1d close > 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below weekly Camarilla S4 AND 1d close < 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price touches weekly Camarilla R3/S3 (profit take) or R5/S5 (stop loss)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong structural levels from higher timeframe
# 1d EMA filter ensures alignment with medium-term trend, reducing counter-trend trades
# High volume threshold (2.0x) filters weak breakouts and ensures conviction
# Works in bull (breakout continuation) and bear (breakdown continuation)

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dEMA50_Volume"
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
    
    # Get weekly data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need sufficient data for weekly pivot (need at least 1 week)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Based on previous week's high, low, close
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # R5 = Close + 2.0 * (High - Low)
    # S5 = Close - 2.0 * (High - Low)
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    close_series_1w = pd.Series(close_1w)
    
    camarilla_r4 = (close_series_1w.shift(1) + 1.5 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    camarilla_s4 = (close_series_1w.shift(1) - 1.5 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    camarilla_r3 = (close_series_1w.shift(1) + 1.0 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    camarilla_s3 = (close_series_1w.shift(1) - 1.0 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    camarilla_r5 = (close_series_1w.shift(1) + 2.0 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    camarilla_s5 = (close_series_1w.shift(1) - 2.0 * (high_series_1w.shift(1) - low_series_1w.shift(1))).values
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r5)
    camarilla_s5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s5)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r5_aligned[i]) or np.isnan(camarilla_s5_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with 1d close > EMA50 and volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and close[i-1] <= camarilla_r4_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with 1d close < EMA50 and volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and close[i-1] >= camarilla_s4_aligned[i-1] and 
                  close[i] < ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: profit take at R3 or stop loss at S5
            if close[i] <= camarilla_r3_aligned[i]:  # Profit take at R3
                signals[i] = 0.0
                position = 0
            elif close[i] <= camarilla_s5_aligned[i]:  # Stop loss at S5
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: profit take at S3 or stop loss at R5
            if close[i] >= camarilla_s3_aligned[i]:  # Profit take at S3
                signals[i] = 0.0
                position = 0
            elif close[i] >= camarilla_r5_aligned[i]:  # Stop loss at R5
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals