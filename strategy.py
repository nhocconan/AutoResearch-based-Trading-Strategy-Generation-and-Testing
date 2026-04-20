#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Camarilla_R3S3_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 48:  # Need at least 2 days of 6h data
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla levels from PREVIOUS day (to avoid look-ahead) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R3 and S3 levels (stronger reversal/breakout zones)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # === 6h: EMA34 for trend filter ===
    close = prices['close'].values
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = close[i]
        ema_val = ema_34[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r3_val) or np.isnan(s3_val) or
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA (uptrend), breaks above R3, volume confirmation
            if (close_val > ema_val and  # Uptrend filter
                close_val > r3_val and   # Break above R3
                vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA (downtrend), breaks below S3, volume confirmation
            elif (close_val < ema_val and  # Downtrend filter
                  close_val < s3_val and   # Break below S3
                  vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below EMA or breaks below S3 (reversal)
            if close_val < ema_val or close_val < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above EMA or breaks above R3 (reversal)
            if close_val > ema_val or close_val > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals