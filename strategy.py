#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume confirmation and 1h RSI filter.
# Long when: Price touches Camarilla S1/S2/S3 AND 1d volume > 1.5x 20-period average AND 1h RSI < 40
# Short when: Price touches Camarilla R1/R2/R3 AND 1d volume > 1.5x 20-period average AND 1h RSI > 60
# Exit when price reaches Camarilla C (central pivot) or RSI crosses 50.
# Uses Camarilla from 1d for structure, volume spike for confirmation, and RSI for timing.
# Designed for 4h timeframe with 20-50 trades/year to avoid fee drag.
# Works in bull markets via reversals at support, in bear via reversals at resistance.
name = "4h_Camarilla_R1S1_Volume_RSI"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4
    # Central pivot: C = (H + L + C) / 3
    
    # Shift by 1 to use previous day's data
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    c = pivot  # central pivot
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma * 1.5)
    
    # Align all 1d data to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    c_aligned = align_htf_to_ltf(prices, df_1d, c)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S1/S2/S3 AND volume spike AND RSI < 40
            long_condition = (
                ((low[i] <= s1_aligned[i]) or (low[i] <= s2_aligned[i]) or (low[i] <= s3_aligned[i])) and
                volume_spike_aligned[i] and
                (rsi[i] < 40)
            )
            # Short: price touches R1/R2/R3 AND volume spike AND RSI > 60
            short_condition = (
                ((high[i] >= r1_aligned[i]) or (high[i] >= r2_aligned[i]) or (high[i] >= r3_aligned[i])) and
                volume_spike_aligned[i] and
                (rsi[i] > 60)
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches central pivot OR RSI > 50
            if (high[i] >= c_aligned[i]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches central pivot OR RSI < 50
            if (low[i] <= c_aligned[i]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals