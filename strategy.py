#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Uses Camarilla pivot levels calculated from 1d high/low/close to identify key S3/R3 levels.
# Long when price breaks above R3 with volume spike and 1d EMA34 up.
# Short when price breaks below S3 with volume spike and 1d EMA34 down.
# Camarilla levels are designed to identify reversal points in ranging markets but also work
# as breakout levels in trending markets. The 1d EMA34 filter ensures we only trade
# in the direction of the higher timeframe trend, reducing whipsaw.
# Volume spike confirms institutional participation in the breakout.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # We only need S3 and R3 for entry
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.full_like(prev_close, np.nan)
    camarilla_R3 = np.full_like(prev_close, np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            continue  # Skip first day as we need previous day's data
        high_val = prev_high[i-1]
        low_val = prev_low[i-1]
        close_val = prev_close[i-1]
        range_val = high_val - low_val
        
        camarilla_S3[i] = close_val - range_val * 1.1 / 4
        camarilla_R3[i] = close_val + range_val * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and 1d EMA34 up
            if (price > camarilla_R3_aligned[i] and 
                vol_spike[i] and price > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and 1d EMA34 down
            elif (price < camarilla_S3_aligned[i] and 
                  vol_spike[i] and price < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) or 1d EMA34 turns down
            if price < camarilla_S3_aligned[i] or price < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or 1d EMA34 turns up
            if price > camarilla_R3_aligned[i] or price > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals