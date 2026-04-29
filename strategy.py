#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot (R3/S3) Breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivots identify key support/resistance levels based on prior day's range.
# Breakout above R3 or below S3 with volume confirmation (>2x 20-period avg) indicates strong momentum.
# 1d EMA34 filter ensures trades align with higher-timeframe trend to avoid counter-trend whipsaws.
# Designed for ~12-25 trades/year on 12h timeframe to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (failed breaks reverse to opposite side).

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
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 12h bar (using previous bar's high/low/close)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4, 
    #            S3 = close - 1.1*(high-low)/4, S4 = close - 1.1*(high-low)/2
    # We use R3/S3 as breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    hl_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * hl_range / 4
    camarilla_s3 = prev_close - 1.1 * hl_range / 4
    
    # Calculate 20-period average volume for spike confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        
        # Handle exits: reverse position if price breaks opposite Camarilla level
        if position == 1:  # Long position
            # Exit and reverse to short if price breaks below S3 with volume
            if curr_low < curr_s3 and curr_volume > 2.0 * curr_vol_ma:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit and reverse to long if price breaks above R3 with volume
            if curr_high > curr_r3 and curr_volume > 2.0 * curr_vol_ma:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume in uptrend (price > 1d EMA34)
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema34_1d:
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below S3 with volume in downtrend (price < 1d EMA34)
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema34_1d:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals