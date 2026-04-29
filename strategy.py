#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 4h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 4h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 OR price crosses 4h EMA50 OR ATR stoploss (2.0)
# Uses 4h HTF for trend (more stable than 1h) and 1d HTF for Camarilla pivots (structure)
# Session filter: 08-20 UTC to avoid low-volume Asian session noise
# Position size: 0.20 discrete to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_h3 = pivot + range_1d * 1.1 / 4
    camarilla_l3 = pivot - range_1d * 1.1 / 4
    camarilla_r3 = pivot + range_1d * 1.1 / 2
    camarilla_s3 = pivot - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        
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
            # Exit conditions: Close below H3 OR price below 4h EMA50 OR stoploss hit
            if curr_close < curr_h3 or curr_close < curr_ema_4h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above L3 OR price above 4h EMA50 OR stoploss hit
            if curr_close > curr_l3 or curr_close > curr_ema_4h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: Close > R3 AND price > 4h EMA50 AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_4h and
                vol_spike):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: Close < S3 AND price < 4h EMA50 AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_4h and
                  vol_spike):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals