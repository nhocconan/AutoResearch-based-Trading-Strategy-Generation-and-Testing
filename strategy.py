#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long: Close breaks above Camarilla R3 AND 12h EMA34 rising (price > EMA34) AND volume > 2.0x 20-period MA
# Short: Close breaks below Camarilla S3 AND 12h EMA34 falling (price < EMA34) AND volume > 2.0x 20-period MA
# Exit: Opposite Camarilla breakout (R4/S4) or EMA34 cross reversal or volume drops below 1.5x MA
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 12h EMA34 filters for trend alignment;
# Volume spike confirms institutional participation. Works in bull via longs and bear via shorts.

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot and EMA34
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day's HLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    #          S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    df_12h_close = df_12h['close'].values
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    
    # Previous bar's HLC for Camarilla calculation
    prev_close = np.roll(df_12h_close, 1)
    prev_high = np.roll(df_12h_high, 1)
    prev_low = np.roll(df_12h_low, 1)
    prev_close[0] = df_12h_close[0]  # First value
    prev_high[0] = df_12h_high[0]
    prev_low[0] = df_12h_low[0]
    
    # Camarilla levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Calculate 12h EMA34
    ema_34 = pd.Series(df_12h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    volume_normal = volume > (1.5 * vol_ma_20)  # For exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        vol_normal = volume_normal[i]
        
        # Determine trend via EMA34
        is_uptrend = close_val > ema_val
        is_downtrend = close_val < ema_val
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND uptrend AND volume spike
            if close_val > r3 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND downtrend AND volume spike
            elif close_val < s3 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks above Camarilla R4 (profit-taking) OR 
            #           Close breaks below Camarilla S3 (reversal) OR
            #           EMA34 cross down (trend change) OR
            #           Volume drops below normal
            if close_val > r4 or close_val < s3 or not is_uptrend or not vol_normal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks below Camarilla S4 (profit-taking) OR
            #           Close breaks above Camarilla R3 (reversal) OR
            #           EMA34 cross up (trend change) OR
            #           Volume drops below normal
            if close_val < s4 or close_val > r3 or not is_downtrend or not vol_normal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals