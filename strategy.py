#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams %R extreme filter and volume spike confirmation.
# Uses 1d HTF for Williams %R (overbought/oversold) and Camarilla levels from previous day to avoid look-ahead.
# Long when price breaks above Camarilla R3 AND Williams %R < -80 (oversold) AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND Williams %R > -20 (overbought) AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla H3/L3 midline.
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Williams %R helps avoid buying into overbought conditions or selling into oversold conditions, improving win rate in both bull and bear markets.

name = "6h_Camarilla_R3S3_1dWilliamsR_Extreme_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Use previous day's OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: based on previous day's range
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 6
    camarilla_l3 = prev_close - rang * 1.1 / 6
    camarilla_h3_l3_mid = (camarilla_h3 + camarilla_l3) / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_l3_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_l3_mid)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_l3_mid_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla H3, oversold (Williams %R < -80), volume confirmation
            if (curr_high > camarilla_h3_aligned[i] and 
                williams_r_aligned[i] < -80 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Camarilla S3, overbought (Williams %R > -20), volume confirmation
            elif (curr_low < camarilla_l3_aligned[i] and 
                  williams_r_aligned[i] > -20 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close < camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close > camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals