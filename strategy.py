#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for stronger trend filter to reduce whipsaw in ranging markets
# Camarilla R3/S3 levels from previous 6h act as breakout levels (tighter, more frequent signals)
# Volume spike confirms breakout validity with 2.0x 20-period average
# Designed for moderate trade frequency (target: 50-150 total over 4 years) to balance opportunity and fee drag
# Works in bull markets via trend-following breaks and in bear markets via avoidance of counter-trend trades

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Need at least 1 previous 6h bar for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 6h bar
        # Camarilla R3 = close + 1.1*(high-low)/2
        # Camarilla S3 = close - 1.1*(high-low)/2
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_range = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * camarilla_range / 2
        camarilla_s3 = prev_close - 1.1 * camarilla_range / 2
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        vol_spike = volume[i] > 2.0 * vol_ma_20
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Exit conditions: price below Camarilla S3 OR price below 1d EMA34
            if curr_close < camarilla_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price above Camarilla R3 OR price above 1d EMA34
            if curr_close > camarilla_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if curr_close > camarilla_r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif curr_close < camarilla_s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals