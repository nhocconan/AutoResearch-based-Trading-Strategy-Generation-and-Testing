#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level in 1d uptrend with volume spike (>1.5x 20-period volume MA).
# Short when price breaks below Camarilla S3 level in 1d downtrend with volume spike.
# Camarilla levels provide high-probability reversal points derived from prior day's range.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h data (based on prior completed 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous bar's high, low, close
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We shift by 1 to use only completed bars (no look-ahead)
    shifted_high = np.roll(high_12h, 1)
    shifted_low = np.roll(low_12h, 1)
    shifted_close = np.roll(close_12h, 1)
    shifted_high[0] = np.nan  # First bar has no prior bar
    shifted_low[0] = np.nan
    shifted_close[0] = np.nan
    
    camarilla_r3 = shifted_close + (shifted_high - shifted_low) * 1.1 / 2
    camarilla_s3 = shifted_close - (shifted_high - shifted_low) * 1.1 / 2
    
    # Align Camarilla levels to lower timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close_val > camarilla_r3_aligned[i] and close_val > ema_34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close_val < camarilla_s3_aligned[i] and close_val < ema_34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 OR 1d trend changes to downtrend
            if close_val < camarilla_s3_aligned[i] or close_val < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 OR 1d trend changes to uptrend
            if close_val > camarilla_r3_aligned[i] or close_val > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals