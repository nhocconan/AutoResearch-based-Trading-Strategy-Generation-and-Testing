#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance and Support levels
    R1 = pivot + (range_ * 1.0 / 6)
    R2 = pivot + (range_ * 2.0 / 6)
    R3 = pivot + (range_ * 3.0 / 6)
    R4 = pivot + (range_ * 4.0 / 6)
    S1 = pivot - (range_ * 1.0 / 6)
    S2 = pivot - (range_ * 2.0 / 6)
    S3 = pivot - (range_ * 3.0 / 6)
    S4 = pivot - (range_ * 4.0 / 6)
    
    # Daily trend filter: EMA34 on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5 * 20-period SMA of volume
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_sma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + in upper half + volume confirmation + daily uptrend
            if (price > R1[i] and  # Breakout above R1
                price > (R1[i] + S1[i]) / 2 and  # In upper half of R1-S1 range
                volume_filter[i] and  # Volume confirmation
                price > ema34_1d_aligned[i]):  # Daily uptrend
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 + in lower half + volume confirmation + daily downtrend
            elif (price < S1[i] and  # Breakdown below S1
                  price < (R1[i] + S1[i]) / 2 and  # In lower half of R1-S1 range
                  volume_filter[i] and  # Volume confirmation
                  price < ema34_1d_aligned[i]):  # Daily downtrend
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to pivot or daily trend fails
            if (price <= pivot[i] or  # Return to pivot
                price < ema34_1d_aligned[i]):  # Daily trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or daily trend fails
            if (price >= pivot[i] or  # Return to pivot
                price > ema34_1d_aligned[i]):  # Daily trend fail
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals