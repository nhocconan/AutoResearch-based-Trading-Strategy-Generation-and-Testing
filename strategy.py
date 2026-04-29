#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA50 trend + volume spike
# Uses 4h for signal direction (EMA50 trend), 1d for Camarilla levels (key S/R),
# and 1h only for entry timing precision. Volume confirms breakout strength.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
# Works in bull/bear: EMA50 filter avoids counter-trend trades, Camarilla levels
# adapt to volatility, volume spike filters weak breakouts.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dLevel_Volume_v1"
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
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Camarilla levels from previous day
    prev_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    camarilla_r3_1d = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)
    camarilla_s3_1d = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)
    
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_1d_aligned[i]
        curr_s3 = camarilla_s3_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit: price drops below 4h EMA50 (trend change) OR breaks Camarilla S3 (failed breakout)
            if curr_close < curr_ema_50_4h or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above 4h EMA50 (trend change) OR breaks Camarilla R3 (failed breakout)
            if curr_close > curr_ema_50_4h or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Only enter during session
            if not session_filter[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Camarilla R3 + above 4h EMA50 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_4h and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 + below 4h EMA50 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_4h and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals