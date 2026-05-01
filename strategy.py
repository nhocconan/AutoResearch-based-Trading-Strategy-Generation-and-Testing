#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 12h EMA50 is rising AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND 12h EMA50 is falling AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels from 1d provide institutional support/resistance. 12h EMA50 ensures alignment with medium-term trend.
# Volume spike threshold set to 2.0x to avoid choppy market noise and ensure institutional participation.
# Primary timeframe: 6h, HTF: 12h for trend and 1d for Camarilla levels.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_12h_aligned, prepend=ema_50_12h_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Actually standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's range to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(ema_rising[i]) or np.isnan(ema_falling[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above Camarilla R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below Camarilla S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R3 AND 12h EMA50 rising AND volume confirmation
            if (breakout_up and 
                ema_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S3 AND 12h EMA50 falling AND volume confirmation
            elif (breakout_down and 
                  ema_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 (stoploss) OR 12h EMA50 turns falling
            if (curr_low < camarilla_s3_aligned[i] or 
                not ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 (stoploss) OR 12h EMA50 turns rising
            if (curr_high > camarilla_r3_aligned[i] or 
                ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals