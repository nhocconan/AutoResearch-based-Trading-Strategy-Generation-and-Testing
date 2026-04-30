#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla pivot levels from 1d to identify key support/resistance, requires breakout in direction of 1d EMA34 trend.
# Designed for low trade frequency (~75-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by only taking breakouts aligned with higher timeframe trend.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend and pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + hl_range * 1.1 / 4
    camarilla_s3 = df_1d['close'] - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Volume confirmation: volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 1d EMA34 uptrend, volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 1d EMA34 downtrend, volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or reverses below entry
            if curr_close < curr_s3 or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or reverses above entry
            if curr_close > curr_r3 or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals