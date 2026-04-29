#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout with 1d EMA50 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R1 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S1 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (PP) or opposite level (S1 for long, R1 for short)
# Uses discrete position sizing (0.30) to balance return and fee drag. Target: 20-50 trades/year on 4h timeframe.
# Camarilla R1/S1 are core support/resistance levels. 1d EMA50 filters counter-trend moves,
# volume confirmation ensures breakout strength. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: using tighter R1/S1 levels on 4h timeframe with volume confirmation.

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous bar's range
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12
    # PP = (high+low+close)/3
    # S1 = close - (high-low)*1.1/12
    # S2 = close - (high-low)*1.1/6
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    
    # Calculate for each 1d bar (using previous bar's data to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first bar to NaN (no previous bar)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r1 = camarilla_pp + camarilla_range * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - camarilla_range * 1.1 / 12.0
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_pp = camarilla_pp_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot (PP) or breaks below S1
            if curr_low <= curr_pp or curr_close <= curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot (PP) or breaks above R1
            if curr_high >= curr_pp or curr_close >= curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND price > 1d EMA50 AND volume confirmation
            if curr_high > curr_r1 and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S1 AND price < 1d EMA50 AND volume confirmation
            elif curr_low < curr_s1 and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals