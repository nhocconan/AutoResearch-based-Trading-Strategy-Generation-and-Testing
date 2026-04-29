#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume spike
# Short when price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume spike
# Exit on opposite Camarilla break (R3/S3) or trend reversal
# Camarilla provides clear support/resistance, 1d EMA34 filters for higher timeframe trend,
# volume confirmation ensures momentum validity
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate previous day's Camarilla levels (using prior 1d bar)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # We use the completed 1d bar's OHLC to calculate levels for current 4h period
    prev_close_1d = df_1d['close'].shift(1).values  # Previous day's close
    prev_high_1d = df_1d['high'].shift(1).values    # Previous day's high
    prev_low_1d = df_1d['low'].shift(1).values      # Previous day's low
    
    # Calculate Camarilla levels for previous day
    camarilla_r3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_s3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (constant throughout the day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 1d EMA34 warmup (Camarilla levels available from first 1d bar)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR 1d EMA34 downtrend
            if curr_close < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR 1d EMA34 uptrend
            if curr_close > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1d and  # price above 1d EMA34 for uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1d and  # price below 1d EMA34 for downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals