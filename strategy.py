#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d timeframe for institutional breakout levels
# Requires 1d EMA34 trend alignment to avoid counter-trend trades
# Volume confirmation (>2.0x 20-period average) filters false breakouts
# Fixed 0.30 position size with ATR(14) trailing stop (2.5x ATR) for risk management
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# BTC/ETH focused: Camarilla pivots work across market regimes with volume confirmation

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range / 2.0
    s3_level = close_1d - 1.1 * camarilla_range / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Calculate ATR(14) for trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14)  # EMA34, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema = ema_34_1d_aligned[i]
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
                signals[i] = 0.30
                
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
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema:
                signals[i] = 0.30
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema:
                signals[i] = -0.30
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals