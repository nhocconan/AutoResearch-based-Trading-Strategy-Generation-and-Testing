#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (S1/S3/R1/R3) breakout with 12h trend filter and volume spike.
# Uses Camarilla pivot levels from daily timeframe (more robust than intraday) for institutional support/resistance.
# Long when price breaks above R1 with 12h EMA50 uptrend and volume spike (>2x 20-period average).
# Short when price breaks below S1 with 12h EMA50 downtrend and volume spike.
# Designed for low trade frequency (~20-35/year) to minimize fee decay. Works in both bull and bear markets
# by following higher timeframe trend (12h EMA50) and using institutional pivot levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    r3 = close_1d + (range_1d * 1.2500)
    r1 = close_1d + (range_1d * 1.0833)
    s1 = close_1d - (range_1d * 1.0833)
    s3 = close_1d - (range_1d * 1.2500)
    
    # Calculate 50-period EMA on 12h close for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Camarilla levels to 4h timeframe (waits for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R1 + uptrend + volume spike
            if price > r1_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + downtrend + volume spike
            elif price < s1_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below S1 or trend breaks
                if price < s1_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above R3 or trend breaks
                if price > r3_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1R1_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0