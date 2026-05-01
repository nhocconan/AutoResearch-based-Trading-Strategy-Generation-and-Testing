#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams %R extreme filter and volume confirmation.
# Uses 1d for signal direction (Williams %R < -80 for long setup, > -20 for short setup) and structure (Camarilla levels).
# 6h timeframe only for entry timing precision to avoid overtrading.
# Long when price breaks above Camarilla R3 with 1d Williams %R < -80 (oversold) and volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 with 1d Williams %R > -20 (overbought) and volume confirmation.
# Discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

name = "6h_Camarilla_R3S3_1dWilliamsR_Extreme_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Williams %R and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Resistance and Support levels
    r3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 14  # warmup for Williams %R and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions (using current bar's levels)
        breakout_up = curr_close > r3_aligned[i]  # break above R3
        breakout_down = curr_close < s3_aligned[i]  # break below S3
        
        # Williams %R filter: < -80 = oversold (long setup), > -20 = overbought (short setup)
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND oversold AND volume confirmation
            if (breakout_up and 
                oversold and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND overbought AND volume confirmation
            elif (breakout_down and 
                  overbought and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR Williams %R becomes overbought
            elif (curr_close < r3_aligned[i] and curr_close > s3_aligned[i]) or \
                 williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR Williams %R becomes oversold
            elif (curr_close < r3_aligned[i] and curr_close > s3_aligned[i]) or \
                 williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals