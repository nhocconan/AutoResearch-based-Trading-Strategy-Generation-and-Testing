#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (>2.0x 20-period average)
# Camarilla pivot levels provide institutional support/resistance; breakouts from R3/S3 levels indicate strong momentum
# 1d EMA34 filter ensures trades align with daily trend; volume spike confirms institutional participation
# Works in both bull/bear markets: breakout structure + trend filter adapts to regime changes
# Target: 80-150 total trades over 4 years (20-38/year) on 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # For each bar, calculate Camarilla levels from previous day's data
    # We need to shift by 1 to use previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first bar's previous values to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4.0)  # Resistance 3
    s3 = pivot - (range_val * 1.1 / 4.0)  # Support 3
    r4 = pivot + (range_val * 1.1 / 2.0)  # Resistance 4
    s4 = pivot - (range_val * 1.1 / 2.0)  # Support 4
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1d EMA34 warmup, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_r4 = r4[i]
        curr_s4 = s4[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below S3 level OR price reaches R4 (take profit)
            if curr_close < curr_s3 or curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 level OR price reaches S4 (take profit)
            if curr_close > curr_r3 or curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: breakout above R3 + above 1d EMA34 + volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below S3 + below 1d EMA34 + volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals