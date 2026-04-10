#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Long when price breaks above Camarilla R4 AND 1d volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla S4 AND 1d volume > 1.5x 20-period volume SMA
# - Exit: opposite Camarilla breakout (R3/S3) or volume drops below average
# - Uses 1d Camarilla levels calculated from previous 1d OHLC
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Works in both bull/bear markets by trading breakouts with volume confirmation

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla levels (based on previous day OHLC)
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Align 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x previous 1d volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > camarilla_r4_aligned[i-1]  # Break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i-1]  # Break below S4
        
        # Exit conditions: opposite breakout at R3/S3 or loss of volume confirmation
        exit_long = close[i] < camarilla_s3_aligned[i-1] or not vol_confirm
        exit_short = close[i] > camarilla_r3_aligned[i-1] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals