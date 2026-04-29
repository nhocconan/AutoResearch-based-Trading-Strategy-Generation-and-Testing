#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from previous 4h bar for precise entry/exit levels
# 4h EMA50 provides trend filter to align with higher timeframe momentum
# Volume spike (1.8x 20-period average) confirms breakout validity
# Designed for 1h timeframe with tight entry conditions to minimize fee drag
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate previous 4h Camarilla levels (R3, S3) - requires high, low, close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (delayed by one 4h bar for completed bar)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = np.zeros(n)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_4h_aligned[i]
        curr_s3 = camarilla_s3_4h_aligned[i]
        curr_atr = atr[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 1.0 * ATR below entry (tighter for 1h timeframe)
            stop_price = entry_price - 1.0 * curr_atr
            # Exit conditions: price below S3 OR price below 4h EMA50 OR stoploss hit
            if curr_close < curr_s3 or curr_close < curr_ema_4h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Stoploss: 1.0 * ATR above entry
            stop_price = entry_price + 1.0 * curr_atr
            # Exit conditions: price above R3 OR price above 4h EMA50 OR stoploss hit
            if curr_close > curr_r3 or curr_close > curr_ema_4h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 4h EMA50 AND volume spike
            if curr_high > curr_r3 and curr_close > curr_ema_4h and vol_spike[i]:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below S3 AND price < 4h EMA50 AND volume spike
            elif curr_low < curr_s3 and curr_close < curr_ema_4h and vol_spike[i]:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals