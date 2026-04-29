#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 trend filter and volume spike (>2.0x 20-period average)
# Camarilla pivot levels provide high-probability reversal/breakout zones; breakout direction filtered by 1d EMA34
# Volume spike ensures institutional participation; discrete sizing (0.25) minimizes fee churn
# Works in both bull/bear markets: Camarilla levels adapt to volatility, effective in ranging and trending regimes
# Target: 80-180 total trades over 4 years (20-45/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Calculate Camarilla pivot levels for previous day (using daily OHLC)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    # We use R3/S3 as breakout levels
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1d EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below S3 OR volatility contraction (optional, but we use time-based exit via opposite signal)
            # For simplicity, we exit on opposite signal
            signals[i] = 0.25
            
        elif position == -1:  # Short position
            # Exit: price closes above R3 OR volatility contraction
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