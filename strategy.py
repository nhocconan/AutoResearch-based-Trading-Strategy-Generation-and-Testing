#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses weekly EMA50 for stronger trend filter to reduce whipsaw in ranging markets
# Camarilla R3/S3 from previous daily range act as breakout levels
# Volume spike confirms breakout validity with 2.0x 20-period average
# ATR-based stoploss (2x ATR) manages risk
# Designed for fewer trades (target: 50-150 total over 4 years) to avoid fee drag
# Works in bull markets via trend-following breaks and in bear markets via avoidance of counter-trend trades

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_ATRStop_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        # Need at least 1 previous bar to calculate Camarilla levels
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous bar's daily range
        # Use daily high/low from 1d data (more appropriate than weekly for intraday levels)
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 2:
            signals[i] = 0.0
            continue
            
        # Get previous completed daily bar for Camarilla calculation
        idx_1d = i // (24 * 6)  # approximate daily bar index (6 4h bars per day)
        if idx_1d < 1 or idx_1d >= len(df_1d):
            signals[i] = 0.0
            continue
            
        # Get previous completed daily bar for Camarilla calculation
        daily_high = df_1d['high'].values[idx_1d-1]
        daily_low = df_1d['low'].values[idx_1d-1]
        daily_close = df_1d['close'].values[idx_1d-1]
        daily_range = daily_high - daily_low
        
        if daily_range <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels from daily range
        R3 = daily_close + daily_range * 1.1 / 4
        S3 = daily_close - daily_range * 1.1 / 4
        
        curr_close = close[i]
        curr_ema_1w = ema_50_1w_aligned[i]
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
            # Exit conditions: price below S3 OR price below 1w EMA50 OR stoploss hit
            if curr_close < S3 or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above R3 OR price above 1w EMA50 OR stoploss hit
            if curr_close > R3 or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1w EMA50 AND volume spike
            if curr_close > R3 and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below S3 AND price < 1w EMA50 AND volume spike
            elif curr_close < S3 and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals