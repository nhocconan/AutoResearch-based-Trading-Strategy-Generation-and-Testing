#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and ATR trailing stop
# Uses 12h timeframe with 1d HTF for Camarilla pivot levels (R3/S3) and trend filter (1d EMA34)
# Volume confirmation (>2x 20-period average) filters false breakouts
# ATR(14) trailing stop (2.5x ATR) manages risk with minimal whipsaw
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year) minimizing fee drag
# Works in both bull and bear markets: breakouts capture strong moves, volume confirmation avoids fakeouts,
# ATR stop protects during reversals/whipsaw, EMA trend filter ensures alignment with higher timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATRTrail"
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
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    hl_range = high_1d - low_1d
    r3 = close_1d + 1.1 * hl_range / 2
    s3 = close_1d - 1.1 * hl_range / 2
    
    # Align Camarilla levels to 12h timeframe (delayed until 1d bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for trailing stop on 12h data
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation on 12h data
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14)  # EMA, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle trailing stop and exits
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Trailing stop: price drops 2.5*ATR from highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below S3 level
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Trailing stop: price rises 2.5*ATR from lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above R3 level
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume confirmation and above 1d EMA34 (uptrend)
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short entry: price breaks below S3 with volume confirmation and below 1d EMA34 (downtrend)
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals