#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout + 1d volume spike + 12h EMA50 trend filter
# Camarilla pivot levels from 1d: breakout above R3 or below S3 with volume confirmation
# Trend filter: 12h EMA50 to ensure alignment with intermediate-term trend
# Volume confirmation: 1d volume > 2.0x 20-period average to ensure participation
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla R3/S3 breakouts capture strong momentum moves; volume confirms institutional participation;
# 12h EMA50 filter avoids counter-trend trades in choppy markets

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's high, low, close)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    r4 = prev_close + rang * 1.1 / 2
    s4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d volume spike filter: volume > 2.0x 20-period average
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        volume_spike = df_1d['volume'].values > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 with volume spike AND price > 12h EMA50 (uptrend)
            if close[i] > r3_aligned[i] and volume_spike_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 with volume spike AND price < 12h EMA50 (downtrend)
            elif close[i] < s3_aligned[i] and volume_spike_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below R3 (failed breakout) OR closes below 12h EMA50 (trend change)
            if close[i] < r3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S3 (failed breakdown) OR closes above 12h EMA50 (trend change)
            if close[i] > s3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals