#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) from prior 1d to identify key support/resistance
# Long when price breaks above R3 with volume > 2x 20-period average AND price > 1d EMA34 (uptrend)
# Short when price breaks below S3 with volume > 2x 20-period average AND price < 1d EMA34 (downtrend)
# Exits on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal
# Designed for 15-25 trades/year on 12h timeframe to minimize fee drag while capturing strong moves
# Uses discrete position sizing (0.25) to reduce churn and works in both bull/bear markets via trend filter

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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 1d Camarilla pivot levels (R3, S3, R4, S4)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3_1d = close_1d_arr + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d_arr - 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_r4_1d = close_1d_arr + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_s4_1d = close_1d_arr - 1.1 * (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use prior completed 1d bar's levels)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Calculate 20-period average volume for confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        camarilla_r3 = camarilla_r3_1d_aligned[i]
        camarilla_s3 = camarilla_s3_1d_aligned[i]
        camarilla_r4 = camarilla_r4_1d_aligned[i]
        camarilla_s4 = camarilla_s4_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches S3 (opposite level) or trend reverses (price < EMA34)
            if curr_low <= camarilla_s3 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches R3 (opposite level) or trend reverses (price > EMA34)
            if curr_high >= camarilla_r3 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume confirmation AND uptrend (price > EMA34)
            if vol_confirm and curr_high > camarilla_r3 and curr_close > curr_ema34_1d:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation AND downtrend (price < EMA34)
            elif vol_confirm and curr_low < camarilla_s3 and curr_close < curr_ema34_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals