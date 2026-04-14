#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with 1d trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1d EMA(50) rising AND volume > 1.5x average
# Short when price breaks below Camarilla S3 AND 1d EMA(50) falling AND volume > 1.5x average
# Exit when price crosses back through Camarilla H-L midpoint
# Camarilla levels from 1d provide robust support/resistance; 1d EMA ensures trend alignment; volume confirms strength
# Works in bull/bear by following 1d trend while using 6b for precise entries
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    H_L_mid = (prev_high + prev_low) / 2
    
    # Calculate EMA on 1d (50-period) for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d data to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H_L_mid_aligned = align_htf_to_ltf(prices, df_1d, H_L_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(H_L_mid_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above R3 AND 1d EMA rising AND volume confirmation
            if (high_val > R3_aligned[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below S3 AND 1d EMA falling AND volume confirmation
            elif (low_val < S3_aligned[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below H-L midpoint
            if close_val < H_L_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above H-L midpoint
            if close_val > H_L_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0