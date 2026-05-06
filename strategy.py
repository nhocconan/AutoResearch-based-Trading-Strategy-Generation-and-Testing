#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 level AND 1w close > 1w EMA50 AND volume > 2.0 * avg_volume(50)
# Short when price breaks below 1d Camarilla S3 level AND 1w close < 1w EMA50 AND volume > 2.0 * avg_volume(50)
# Exit when price crosses 1d EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Camarilla provides clear structure with proven breakout edge
# 1w EMA50 filter ensures alignment with higher timeframe trend (works in bull/bear)
# High volume confirmation filters weak breakouts (reduces false signals)
# Works in bull: breaks above R3 in uptrend capture moves
# Works in bear: breaks below S3 in downtrend capture moves
# Range: volume spikes at extremes often precede reversals

name = "12h_1dCamarillaR3S3_1wEMA50Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels based on previous 1d bar
    # Camarilla: R3 = Close + 1.125 * (High - Low), S3 = Close - 1.125 * (High - Low)
    camarilla_r3_1d = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.125 * (high_1d - low_1d)
    
    # Calculate 1d EMA50 for trend filter and exit signal
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Camarilla levels and EMA to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align 1w EMA to 12h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 50-period average volume
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * avg_volume_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 1w EMA50 uptrend and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                close_1w[i] > ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 1w EMA50 downtrend and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  close_1w[i] < ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals