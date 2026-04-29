#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for stronger trend filter to reduce whipsaw in ranging markets
# Camarilla R3/S3 levels from previous 4h act as breakout levels (tighter, more frequent signals)
# Volume spike confirms breakout validity with 2.0x 20-period average
# ATR-based stoploss (2x ATR) manages risk
# Designed for moderate trade frequency (target: 100-200 total over 4 years) to balance opportunity and fee drag
# Works in bull markets via trend-following breaks and in bear markets via avoidance of counter-trend trades

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 1 previous 4h bar for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 4h bar
        # Camarilla R3 = close + 1.1*(high-low)/2
        # Camarilla S3 = close - 1.1*(high-low)/2
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_range = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * camarilla_range / 2
        camarilla_s3 = prev_close - 1.1 * camarilla_range / 2
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below Camarilla S3 OR price below 12h EMA50 OR stoploss hit
            if curr_close < camarilla_s3 or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above Camarilla R3 OR price above 12h EMA50 OR stoploss hit
            if curr_close > camarilla_r3 or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 12h EMA50 AND volume spike
            if curr_close > camarilla_r3 and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 AND price < 12h EMA50 AND volume spike
            elif curr_close < camarilla_s3 and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals