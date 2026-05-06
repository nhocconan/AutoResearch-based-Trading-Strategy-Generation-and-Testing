#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) with 1h EMA trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1h EMA50 > EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 1h EMA50 < EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 1d Camarilla pivot point (PP) or opposite level (S3/R3)
# Uses discrete sizing 0.30 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise intraday support/resistance aligned with institutional order flow
# 1h EMA filter ensures alignment with short-term trend, reducing whipsaw in ranging markets
# High volume confirmation (2.0x) filters weak breakouts and confirms institutional participation
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "4h_1dCamarillaR3S3_1hEMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3, PP)
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1h data ONCE before loop for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1h = df_1h['close'].values
    
    # Calculate 1h EMA50 and EMA200
    close_series_1h = pd.Series(close_1h)
    ema_50_1h = close_series_1h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1h = close_series_1h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1h EMA values to 4h timeframe (wait for completed 1h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    ema_200_aligned = align_htf_to_ltf(prices, df_1h, ema_200_1h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 1h EMA50 > EMA200 and volume confirmation
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 1h EMA50 < EMA200 and volume confirmation
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Camarilla pivot point (PP) or S3 (reversal or profit take)
            if close[i] <= pp_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price touches 1d Camarilla pivot point (PP) or R3 (reversal or profit take)
            if close[i] >= pp_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals