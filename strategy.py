#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 (1d) with volume > 1.5x 20-period average and close > 1d EMA34
# Short when price breaks below Camarilla S3 (1d) with volume > 1.5x 20-period average and close < 1d EMA34
# Exit when price returns to Camarilla Pivot (1d) or volume drops below average
# Uses discrete position sizing (0.25) to minimize fee churn. Camarilla levels from 1d provide
# precise intraday support/resistance. Volume spike confirms institutional interest. 1d EMA34
# filters for higher-timeframe trend alignment. Target: 12-37 trades/year on 12h timeframe.
# Works in both bull and bear markets by only trading strong breakouts with volume confirmation
# and trend alignment, avoiding false breakouts in ranging conditions.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla levels (using previous day's values)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # We use R3 and S3 for breakouts, P for exit
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First bar has no previous day
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    rang = prev_high - prev_low
    camarilla_p = prev_close  # Pivot
    camarilla_r3 = prev_close + rang * 1.1 / 4
    camarilla_s3 = prev_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike filter (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_p = camarilla_p_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price returns to Camarilla Pivot OR volume drops below average
            if curr_close <= curr_p or curr_volume < curr_vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla Pivot OR volume drops below average
            if curr_close >= curr_p or curr_volume < curr_vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_spike = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above Camarilla R3 with volume spike and close > 1d EMA34
            if curr_close > curr_r3 and volume_spike and curr_close > curr_ema34:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 with volume spike and close < 1d EMA34
            elif curr_close < curr_s3 and volume_spike and curr_close < curr_ema34:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals