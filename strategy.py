#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(34) trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 4h EMA(34) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND price < 4h EMA(34) AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.20) to minimize fee drag. Uses 4h/1d for signal direction, 1h only for entry timing.
# Session filter: 08-20 UTC to reduce noise trades. Works in both bull and bear by following HTF trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(34)
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivot levels from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_4h_arr + (high_4h - low_4h) * 1.1 / 2
    camarilla_s3 = close_4h_arr - (high_4h - low_4h) * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long positions
    lowest_since_entry = 0.0   # for short positions
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            # Exit any position outside session
            if position == 1:
                position = 0
                highest_since_entry = 0.0
            elif position == -1:
                position = 0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions:
            # 1. Price breaks below Camarilla S3
            # 2. Price < 4h EMA(34)
            # 3. Trailing stop: price drops 2.5*ATR from highest since entry
            if (curr_close < curr_s3 or 
                curr_close < curr_ema or
                curr_close < highest_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Exit conditions:
            # 1. Price breaks above Camarilla R3
            # 2. Price > 4h EMA(34)
            # 3. Trailing stop: price rises 2.5*ATR from lowest since entry
            if (curr_close > curr_r3 or 
                curr_close > curr_ema or
                curr_close > lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 4h EMA(34) AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_close
            # Short entry: price breaks below Camarilla S3 AND price < 4h EMA(34) AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals