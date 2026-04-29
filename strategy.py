#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout + 1d EMA50 trend filter + volume spike
# Uses 4h for signal direction (Camarilla breakout) and 1d for trend filter (EMA50)
# 1h timeframe provides precise entry timing while keeping trade frequency manageable (target: 15-35 trades/year)
# Volume confirmation reduces false breakouts. Works in bull/bear via trend filter.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    prev_close_4h = np.concatenate([[np.nan], df_4h['close'].values[:-1]])
    prev_high_4h = np.concatenate([[np.nan], df_4h['high'].values[:-1]])
    prev_low_4h = np.concatenate([[np.nan], df_4h['low'].values[:-1]])
    
    camarilla_r3_4h = prev_close_4h + 1.0 * (prev_high_4h - prev_low_4h)
    camarilla_s3_4h = prev_close_4h - 1.0 * (prev_high_4h - prev_low_4h)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_4h_aligned[i]
        curr_s3 = camarilla_s3_4h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_session = session_filter[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price drops below 1d EMA50 (trend change)
            # 2. Price breaks below Camarilla S3 (breakout failed)
            if (curr_close < curr_ema_50_1d or
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above 1d EMA50 (trend change)
            # 2. Price breaks above Camarilla R3 (breakout failed)
            if (curr_close > curr_ema_50_1d or
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Only trade during session and with volume confirmation
            if not (curr_session and curr_volume_confirm):
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Camarilla R3 + above 1d EMA50
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_1d):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: price breaks below Camarilla S3 + below 1d EMA50
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_1d):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals