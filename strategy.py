#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 AND 1d EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 AND 1d EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 12h Camarilla pivot point (PP) or opposite Camarilla level (R3/S3)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 12h Camarilla provides strong intraday support/resistance levels derived from prior day's range
# 1d EMA filter (34/89) ensures alignment with medium-term trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts at R3) and bear (trend continuation breakdowns at S3)

name = "6h_12hCamarilla_R3S3_Breakout_1dEMATrend_Volume"
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
    
    # Get 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 1 prior day for Camarilla calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels from prior 12h bar's range
    # Camarilla formula: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + Range * 1.1/2, S3 = PP - Range * 1.1/2
    # We use the prior completed 12h bar to avoid look-ahead
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    pp_12h = typical_price_12h  # Simplified: using typical price as pivot (standard Camarilla uses prior day's PP)
    camarilla_r3_12h = pp_12h + range_12h * 1.1 / 2.0
    camarilla_s3_12h = pp_12h - range_12h * 1.1 / 2.0
    camarilla_pp_12h = pp_12h  # Pivot point
    
    # Shift by 1 to use prior completed bar's levels (avoid look-ahead)
    camarilla_r3_12h = np.roll(camarilla_r3_12h, 1)
    camarilla_s3_12h = np.roll(camarilla_s3_12h, 1)
    camarilla_pp_12h = np.roll(camarilla_pp_12h, 1)
    camarilla_r3_12h[0] = np.nan  # First value invalid after roll
    camarilla_s3_12h[0] = np.nan
    camarilla_pp_12h[0] = np.nan
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp_12h)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 12h Camarilla PP or falls back below S3 (reversal or profit take)
            if close[i] <= camarilla_pp_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 12h Camarilla PP or rises above R3 (reversal or profit take)
            if close[i] >= camarilla_pp_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals