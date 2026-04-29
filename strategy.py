#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>1.8x 20-period average)
# Camarilla pivots provide mathematically derived support/resistance levels effective in ranging and trending markets
# R3/S3 levels often act as acceleration points - breaks here with volume indicate strong momentum
# 1d EMA34 filters trades to align with higher timeframe trend, reducing counter-trend whipsaws
# Volume confirmation (>1.8x average) ensures breakouts have institutional participation
# Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range
    s3_level = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average (strict threshold for quality)
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below S3 level OR price closes below 1d EMA34
            if curr_close < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 level OR price closes above 1d EMA34
            if curr_close > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 + price above 1d EMA34 + volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + price below 1d EMA34 + volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals