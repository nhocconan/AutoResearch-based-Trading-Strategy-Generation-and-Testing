#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation
# Camarilla levels provide precise intraday support/resistance. Breakout of R3/S3 with
# 1d EMA34 trend filter and volume spike captures strong momentum moves. Works in both
# bull and bear markets by combining price structure (Camarilla) with trend and volume.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
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
    
    start_idx = max(100, 34, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = prices['open'].iloc[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # Need to get previous day's data - we'll use the 1d dataframe
        day_idx = i // 96  # Approximate: 96 4h bars per day (24h * 4)
        if day_idx > 0 and day_idx < len(df_1d):
            prev_high = df_1d['high'].iloc[day_idx-1]
            prev_low = df_1d['low'].iloc[day_idx-1]
            prev_close = df_1d['close'].iloc[day_idx-1]
            diff = prev_high - prev_low
            camarilla_r3 = prev_close + diff * 1.1 / 4
            camarilla_s3 = prev_close - diff * 1.1 / 4
        else:
            camarilla_r3 = curr_close
            camarilla_s3 = curr_close
        
        # Volume confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR close below Camarilla S3
            if curr_close < stop_price or curr_close < camarilla_s3:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop OR close above Camarilla R3
            if curr_close > stop_price or curr_close > camarilla_r3:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: close above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: close below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals