#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for trend direction (more stable than 12h) and requires volume > 2.0x average for confirmation.
# Designed for low trade frequency (~50-100 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by only taking breakouts in the direction of the 1d EMA50 trend.
# Added ATR-based stoploss (2.5x ATR) to manage risk and reduce whipsaws.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period ATR on 4h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels for previous 1d bar (completed)
        if len(df_1d) >= 2:
            # Camarilla levels: based on previous 1d bar's range
            prev_high = df_1d['high'].iloc[-2]
            prev_low = df_1d['low'].iloc[-2]
            prev_close = df_1d['close'].iloc[-2]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r3 = prev_close + (range_val * 1.1 / 4)
            camarilla_s3 = prev_close - (range_val * 1.1 / 4)
            camarilla_r4 = prev_close + (range_val * 1.1 / 2)
            camarilla_s4 = prev_close - (range_val * 1.1 / 2)
            
            # Create arrays aligned to 1d timeframe
            camarilla_r3_1d = np.full(len(df_1d), camarilla_r3)
            camarilla_s3_1d = np.full(len(df_1d), camarilla_s3)
            camarilla_r4_1d = np.full(len(df_1d), camarilla_r4)
            camarilla_s4_1d = np.full(len(df_1d), camarilla_s4)
            
            # Align to 4h timeframe with proper delay (wait for 1d bar to close)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
        else:
            camarilla_r3_aligned = np.full(n, np.nan)
            camarilla_s3_aligned = np.full(n, np.nan)
            camarilla_r4_aligned = np.full(n, np.nan)
            camarilla_s4_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 2.0x 20-period average (stricter)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 1d EMA50 uptrend, volume spike
            if (curr_close > camarilla_r3_aligned[i] and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 1d EMA50 downtrend, volume spike
            elif (curr_close < camarilla_s3_aligned[i] and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3, or ATR stoploss hit
            if curr_close < camarilla_s3_aligned[i] or curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3, or ATR stoploss hit
            if curr_close > camarilla_r3_aligned[i] or curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals