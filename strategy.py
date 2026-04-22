#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Fade with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous day (R3, S3) for mean reversion entries
# Long when price pulls back to S3 in 1d uptrend with volume confirmation
# Short when price pulls back to R3 in 1d downtrend with volume confirmation
# 1d trend filter ensures we trade with higher timeframe momentum, reducing counter-trend losses
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Fading at R3/S3 captures mean reversion within the trend, working in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_R3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_S3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20  # Volume above 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3_1d_aligned[i]) or np.isnan(camarilla_S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to S3 + 1d uptrend + volume confirmation
            if (close[i] <= camarilla_S3_1d_aligned[i] * 1.005 and  # Allow small buffer
                close[i] >= camarilla_S3_1d_aligned[i] * 0.995 and
                close[i] > ema_34_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to R3 + 1d downtrend + volume confirmation
            elif (close[i] <= camarilla_R3_1d_aligned[i] * 1.005 and 
                  close[i] >= camarilla_R3_1d_aligned[i] * 0.995 and
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price moves to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long if price reaches R3 or trend turns down
                if (close[i] >= camarilla_R3_1d_aligned[i] * 0.995 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price reaches S3 or trend turns up
                if (close[i] <= camarilla_S3_1d_aligned[i] * 1.005 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dEMA34_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0