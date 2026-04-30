#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with volume > 2.0x 20-bar average AND close > 1d EMA34.
# Short when price breaks below S3 with volume > 2.0x 20-bar average AND close < 1d EMA34.
# Exit when price reverts to the Camarilla pivot point (PP) or opposite S1/R1 level.
# Camarilla levels provide precise intraday support/resistance derived from prior day's range.
# 1d EMA34 filters for dominant daily trend to avoid counter-trend breakouts.
# Volume confirmation ensures institutional participation in the breakout.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # Camarilla: PP = (H1 + L1 + C1) / 3
    # R4 = C1 + (H1-L1)*1.5/2, R3 = C1 + (H1-L1)*1.25/2, R2 = C1 + (H1-L1)*1.1/2, R1 = C1 + (H1-L1)*1.05/2
    # S1 = C1 - (H1-L1)*1.05/2, S2 = C1 - (H1-L1)*1.1/2, S3 = C1 - (H1-L1)*1.25/2, S4 = C1 - (H1-L1)*1.5/2
    # We use R3/S3 for breakout and PP for exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar will have invalid prior day data (will be handled by min_periods later)
    
    # Calculate Camarilla levels
    PP = (prev_high + prev_low + prev_close) / 3.0
    R3 = prev_close + (prev_high - prev_low) * 1.25 / 2.0
    S3 = prev_close - (prev_high - prev_low) * 1.25 / 2.0
    # Also calculate PP for exit
    
    # Align Camarilla levels to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3, uptrend (close > 1d EMA34), volume confirmation
            if (curr_close > R3_aligned[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3, downtrend (close < 1d EMA34), volume confirmation
            elif (curr_close < S3_aligned[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Price reverts to pivot point (PP) or below S1
            # S1 = C - (H-L)*1.05/2, but we use PP as simpler exit
            if curr_close <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Price reverts to pivot point (PP) or above R1
            if curr_close >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals