#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R1 with 4h EMA50 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below S1 with 4h EMA50 downtrend and volume > 2.0x 20-bar average.
# Uses session filter (08-20 UTC) to reduce noise trades. Discrete sizing 0.20 to minimize fee churn.
# Designed for 1h timeframe with HTF direction from 4h/1d to avoid overtrading.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 calculation
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        
        # Volume confirmation: current 1h volume > 2.0x 20-bar average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 AND price > 4h EMA50 AND volume confirmation
            if (curr_close > curr_r1 and 
                curr_close > curr_ema_50_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND price < 4h EMA50 AND volume confirmation
            elif (curr_close < curr_s1 and 
                  curr_close < curr_ema_50_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S1 (reversal) OR price < 4h EMA50 (trend violation)
            if (curr_close < curr_s1 or 
                curr_close < curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above R1 (reversal) OR price > 4h EMA50 (trend violation)
            if (curr_close > curr_r1 or 
                curr_close > curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals