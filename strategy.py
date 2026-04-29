#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA34 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R1 AND price > 4h EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (PP) or opposite level (S1 for long, R1 for short)
# Uses 4h/1d for signal direction (trend + structure), 1h only for entry timing precision.
# Session filter: 08-20 UTC to reduce noise trades. Discrete position sizing (0.20) to minimize fee drag.
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation and EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
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
    
    # Calculate for each 4h bar (using previous bar's data to avoid look-ahead)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    
    # Set first bar to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_r1 = camarilla_pp + camarilla_range * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - camarilla_range * 1.1 / 12.0
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 4h indicators to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_pp = camarilla_pp_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema34_4h = ema_34_4h_aligned[i]
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
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot (PP) or breaks above R1
            if curr_high >= curr_pp or curr_close >= curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND price > 4h EMA34 AND volume confirmation
            if curr_high > curr_r1 and curr_close > curr_ema34_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND price < 4h EMA34 AND volume confirmation
            elif curr_low < curr_s1 and curr_close < curr_ema34_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals