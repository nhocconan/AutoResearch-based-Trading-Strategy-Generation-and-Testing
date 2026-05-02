#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Uses 6h timeframe for signal generation with Camarilla pivot breakouts at R3/S3 levels
# 12h EMA34 provides multi-timeframe trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Designed for moderate trade frequency (target: 50-150 total trades over 4 years)
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Based on proven Camarilla breakout patterns with proper risk management

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike_v1"
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
    open_price = prices['open'].values
    
    # Load 12h data ONCE before loop for Camarilla pivots and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous completed 12h bar to avoid look-ahead
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r3 = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_r4 = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_s4 = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for 12h bar to close)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 12h EMA34 + volume confirm
            if close[i] > r3_12h_aligned[i] and close[i] > ema_34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 12h EMA34 + volume confirm
            elif close[i] < s3_12h_aligned[i] and close[i] < ema_34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals