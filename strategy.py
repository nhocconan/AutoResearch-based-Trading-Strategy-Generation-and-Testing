#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1w EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 1w EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 1d Camarilla midpoint (R3/S3 midpoint) or opposite level (S3/R3)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Camarilla R3/S3 provides strong intraday support/resistance levels proven to work on ETH/SOL
# 1w EMA filter ensures alignment with long-term trend, reducing counter-trend trades in bear markets
# Volume confirmation with higher threshold (2.0x) filters weak breakouts and reduces overtrading
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns + mean reversion at extremes)

name = "4h_1dCamarilla_R3S3_Breakout_1wEMATrend_Volume"
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
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation (previous day's OHLC)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # Pivot = (high + low + close) / 3
    # We use R3 and S3 as breakout levels
    prev_high = np.roll(high_1d, 1)  # Previous day's high
    prev_low = np.roll(low_1d, 1)    # Previous day's low
    prev_close = np.roll(close_1d, 1) # Previous day's close
    
    # Set first value to NaN since no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0  # Midpoint between R3 and S3
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA89
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 4h timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 1w EMA34 > EMA89 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 1w EMA34 < EMA89 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Camarilla midpoint or S3 (reversal or profit take)
            if close[i] <= camarilla_mid_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Camarilla midpoint or R3 (reversal or profit take)
            if close[i] >= camarilla_mid_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals