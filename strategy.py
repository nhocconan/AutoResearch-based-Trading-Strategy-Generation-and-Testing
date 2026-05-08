#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 on 4h, 12h EMA50 rising, volume > 1.5x average
# Short when price breaks below S3 on 4h, 12h EMA50 falling, volume > 1.5x average
# Camarilla levels provide strong support/resistance; EMA50 filters whipsaws; volume confirms momentum
# Targets 75-200 total trades over 4 years (19-50/year) for low fee drag and high win rate

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    HLC_diff = high_4h - low_4h
    camarilla_r3 = close_4h + (HLC_diff * 1.1 / 4)
    camarilla_s3 = close_4h - (HLC_diff * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3, 12h uptrend, volume spike
            if high_val > r3_val and ema50_12h_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 12h downtrend, volume spike
            elif low_val < s3_val and ema50_12h_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 12h trend down
            if low_val < s3_val or ema50_12h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or 12h trend up
            if high_val > r3_val or ema50_12h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals