#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 Breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels derived from prior day's range
# R3/S3 represent strong breakout levels - price closing beyond these indicates institutional interest
# Combined with 1w EMA34 for higher timeframe trend alignment and volume confirmation (>1.8x 20-period average)
# Designed to capture sustained moves in both bull and bear markets with tight entry conditions
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day's data for pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get previous day's OHLC for Camarilla pivot calculation
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla R3 and S3 levels
        r3 = pivot + range_val * 1.1 / 4.0
        s3 = pivot - range_val * 1.1 / 4.0
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        
        # Calculate 20-period average volume for confirmation
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_20 = 0
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below R3 (failed breakout) OR below 1w EMA34
            if curr_close < r3 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 (failed breakdown) OR above 1w EMA34
            if curr_close > s3 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price closes above R3 + above 1w EMA34 + volume confirmation
            if (curr_close > r3 and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price closes below S3 + below 1w EMA34 + volume confirmation
            elif (curr_close < s3 and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals