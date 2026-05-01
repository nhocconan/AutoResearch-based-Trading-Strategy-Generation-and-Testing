#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND 4h close > EMA50 AND volume > 1.8x 20-bar average.
# Short when price breaks below S3 AND 4h close < EMA50 AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.20 to manage drawdown. Target: 60-150 total trades over 4 years (15-37/year).
# Primary timeframe: 1h, HTF: 4h for trend filter and Camarilla levels (from 1d).
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load 4h data ONCE before loop for EMA trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 4h EMA50 calculation
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 4h close aligned for trend bias
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate Camarilla pivot levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) to use previous completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3/S3
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 1h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_aligned[i]) or np.isnan(close_4h_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > r3_aligned[i]  # break above R3
        breakout_down = curr_low < s3_aligned[i]  # break below S3
        
        # Trend filter: use 4h close vs its EMA50 for bias
        bullish_bias = close_4h_aligned[i] > ema_aligned[i]  # 4h close above its EMA50 = bullish
        bearish_bias = close_4h_aligned[i] < ema_aligned[i]  # 4h close below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bullish bias AND volume confirmation
            if (breakout_up and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND bearish bias AND volume confirmation
            elif (breakout_down and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR bearish bias (trend change)
            if (curr_low < s3_aligned[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR bullish bias (trend change)
            if (curr_high > r3_aligned[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals