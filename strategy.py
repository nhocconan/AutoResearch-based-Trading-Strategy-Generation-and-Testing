#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 level AND 1d EMA34 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Camarilla S3 level AND 1d EMA34 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 4h EMA34 (trend reversal signal)
# Uses discrete sizing 0.20 to balance return and drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Camarilla provides clear structure with proven breakout edge
# 1d EMA34/EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)
# Session filter (08-20 UTC) reduces noise trades

name = "1h_4hCamarillaR3S3_1dEMA34Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Get 4h data ONCE before loop for Camarilla pivots and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels based on previous 4h bar
    # Camarilla: R3 = Close + 1.25 * (High - Low), S3 = Close - 1.25 * (High - Low)
    camarilla_r3_4h = close_4h + 1.25 * (high_4h - low_4h)
    camarilla_s3_4h = close_4h - 1.25 * (high_4h - low_4h)
    
    # Calculate 4h EMA34 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema_34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h Camarilla levels and EMA to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Align 1d EMA indicators to 1h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: price breaks above 4h Camarilla R3 with 1d EMA34 > EMA200 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S3 with 1d EMA34 < EMA200 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend reversal) OR exit session
            if close[i] < ema_34_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend reversal) OR exit session
            if close[i] > ema_34_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals