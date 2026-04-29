#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 AND price > 12h EMA(50) AND volume > 2.0x 20-period average
# Short when price breaks below 4h Camarilla S3 AND price < 12h EMA(50) AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Based on proven pattern: Camarilla breakouts with volume and trend filters show strong test performance.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Camarilla levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels calculation
    # Range = high - low of previous day (using 4h data, approximate with 6-period lookback for daily range)
    # For 4h timeframe, use 6 periods back to approximate daily range (6*4h = 24h)
    lookback = 6
    if len(high_4h) >= lookback + 1:
        prev_high = np.maximum.accumulate(high_4h)[lookback:]  # rolling max of high
        prev_low = np.minimum.accumulate(low_4h)[lookback:]    # rolling min of low
        prev_close = close_4h[lookback:]
        
        # Align arrays to same length
        min_len = min(len(prev_high), len(prev_low), len(prev_close), len(high_4h[lookback:]), len(low_4h[lookback:]), len(close_4h[lookback:]))
        if min_len > 0:
            prev_high = prev_high[-min_len:]
            prev_low = prev_low[-min_len:]
            prev_close = prev_close[-min_len:]
            curr_high = high_4h[lookback:lookback+min_len]
            curr_low = low_4h[lookback:lookback+min_len]
            curr_close = close_4h[lookback:lookback+min_len]
            
            # Calculate pivot and ranges
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            
            # Camarilla levels
            R3 = pivot + (range_val * 1.1 / 4.0)
            S3 = pivot - (range_val * 1.1 / 4.0)
            
            # Align to full length
            R3_full = np.full(len(high_4h), np.nan)
            S3_full = np.full(len(high_4h), np.nan)
            R3_full[lookback:lookback+min_len] = R3
            S3_full[lookback:lookback+min_len] = S3
            
            # Forward fill to handle NaN values
            R3_series = pd.Series(R3_full)
            S3_series = pd.Series(S3_full)
            R3_filled = R3_series.ffill().bfill().values
            S3_filled = S3_series.ffill().bfill().values
        else:
            R3_filled = np.full(len(high_4h), np.nan)
            S3_filled = np.full(len(high_4h), np.nan)
    else:
        R3_filled = np.full(len(high_4h), np.nan)
        S3_filled = np.full(len(high_4h), np.nan)
    
    camarilla_R3 = align_htf_to_ltf(prices, df_4h, R3_filled)
    camarilla_S3 = align_htf_to_ltf(prices, df_4h, S3_filled)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_R3 = camarilla_R3[i]
        curr_S3 = camarilla_S3[i]
        
        # Skip if Camarilla levels not available
        if np.isnan(curr_R3) or np.isnan(curr_S3):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price < 12h EMA(50)
            if curr_close < curr_S3 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price > 12h EMA(50)
            if curr_close > curr_R3 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 12h EMA(50) AND volume spike
            if (curr_close > curr_R3 and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 12h EMA(50) AND volume spike
            elif (curr_close < curr_S3 and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals