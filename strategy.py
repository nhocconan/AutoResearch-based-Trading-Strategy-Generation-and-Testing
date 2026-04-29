#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide institutional support/resistance; R3/S3 are strong breakout levels
# Combined with 1d EMA34 for higher timeframe trend alignment and volume confirmation (>2.0x 20-period average)
# Designed to capture sustained moves in both bull and bear markets while avoiding choppy conditions
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 12h timeframe using previous bar's OHLC
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # 1d EMA34 warmup, need previous bar for pivot
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Use previous bar's OHLC to calculate Camarilla levels (no look-ahead)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla R3 and S3 levels
        r3 = pivot + (range_val * 1.1 / 4.0)
        s3 = pivot - (range_val * 1.1 / 4.0)
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            vol_confirm = curr_volume > 2.0 * vol_ma_20
        else:
            vol_confirm = False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below R3 OR below 1d EMA34
            if curr_close < r3 or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR above 1d EMA34
            if curr_close > s3 or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 + above 1d EMA34 + volume confirmation
            if (curr_close > r3 and 
                curr_close > ema_34_1d_aligned[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below 1d EMA34 + volume confirmation
            elif (curr_close < s3 and 
                  curr_close < ema_34_1d_aligned[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals