#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 Breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior 1d period: long on break above R4 in uptrend, short on break below S4 in downtrend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Designed for 4h timeframe to capture medium-term swings with controlled trade frequency (~20-40 trades/year)
# Works in both bull and bear markets by aligning with 1d trend (EMA34) to avoid counter-trend trades

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # R4 = close + 1.5*(high-low)*1.1/2, S4 = close - 1.5*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1
    r4 = close_1d + camarilla_range * 1.5 / 2
    s4 = close_1d - camarilla_range * 1.5 / 2
    
    # Align Camarilla levels to 4h timeframe (delayed by one 1d bar for look-ahead avoidance)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: reverse signal on opposite Camarilla level break or trend change
        if position == 1:  # Long position
            # Exit: price breaks below S4 or trend turns down (price < EMA34)
            if curr_low < curr_s4 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R4 or trend turns up (price > EMA34)
            if curr_high > curr_r4 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above R4 in uptrend (price > EMA34)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > curr_r4:  # Break above R4 level
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below S4 in downtrend (price < EMA34)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < curr_s4:  # Break below S4 level
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals