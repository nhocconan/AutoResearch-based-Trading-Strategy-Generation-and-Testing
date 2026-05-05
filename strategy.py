#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d Volume Spike and 6h EMA34 Trend Filter
# Long when: price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 6h close > 6h EMA34
# Short when: price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 6h close < 6h EMA34
# Exit when price returns to Camarilla Pivot Point (mean reversion)
# Camarilla levels provide high-probability reversal points from 1d OHLC
# Volume spike confirms institutional participation
# 6h EMA34 filter ensures alignment with intermediate timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25 to minimize fee churn

name = "4h_Camarilla_R3S3_Breakout_1dVolume_6hTrend"
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
    
    # Get 1d data ONCE before loop for Camarilla levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 6h data ONCE before loop for EMA34 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h EMA(34) for trend filter
    ema_34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_6h, ema_34_6h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2, PP = (H+L+C)/3
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 + volume spike + 6h uptrend
            if (close[i] > r3_aligned[i] and 
                volume_1d[i] > 2.0 * vol_ma_aligned[i] and 
                close_6h[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + volume spike + 6h downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_1d[i] > 2.0 * vol_ma_aligned[i] and 
                  close_6h[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot point (mean reversion)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot point (mean reversion)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals