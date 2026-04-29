#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h Camarilla pivot levels (R3/S3) for institutional breakout structure
# 1d EMA50 provides strong HTF trend filter to align with primary trend direction
# Volume spike (2.0x 20-period average) confirms breakout validity with institutional participation
# Designed for low trade frequency (target: 15-37 trades/year) to minimize fee drag on 1h timeframe
# Works in bull markets via long signals when price breaks above R3 with HTF uptrend
# Works in bear markets via short signals when price breaks below S3 with HTF downtrend
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels from previous 4h candle
    df_4h_close = df_4h['close'].values
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    
    prev_4h_high = np.concatenate([[df_4h_high[0]], df_4h_high[:-1]])
    prev_4h_low = np.concatenate([[df_4h_low[0]], df_4h_low[:-1]])
    prev_4h_close = np.concatenate([[df_4h_close[0]], df_4h_close[:-1]])
    
    camarilla_range_4h = prev_4h_high - prev_4h_low
    r3_4h = prev_4h_close + (camarilla_range_4h * 1.1 / 4.0)
    s3_4h = prev_4h_close - (camarilla_range_4h * 1.1 / 4.0)
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period on 1h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3_4h = r3_4h_aligned[i]
        curr_s3_4h = s3_4h_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below R3 (failed breakout)
            if curr_close < stop_price or curr_close < curr_r3_4h:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop OR price breaks above S3 (failed breakout)
            if curr_close > stop_price or curr_close > curr_s3_4h:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: Price breaks above R3 AND price > 1d EMA50 AND volume spike
            if curr_close > curr_r3_4h and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Price breaks below S3 AND price < 1d EMA50 AND volume spike
            elif curr_close < curr_s3_4h and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals