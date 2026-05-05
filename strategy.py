#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + 12h Volume Spike + 1d Camarilla Pivot Breakout
# Long when: 6h BB width < 20th percentile (squeeze) AND 12h volume > 2.0x 20-period MA AND price breaks above 1d Camarilla R3
# Short when: 6h BB width < 20th percentile (squeeze) AND 12h volume > 2.0x 20-period MA AND price breaks below 1d Camarilla S3
# Exit when: price returns to 6h BB middle (20-period SMA) OR BB width expands above 50th percentile
# Uses volatility contraction (squeeze) for low-risk entries, volume for conviction, Camarilla for institutional levels
# Timeframe: 6h, HTF: 12h for volume, 1d for pivots. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBSqueeze_12hVol_1dCamarilla"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2)
    if len(close) >= 20:
        bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_width = bb_upper - bb_lower
    else:
        bb_ma = np.full(n, np.nan)
        bb_std = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # BB width percentile (20th for squeeze entry, 50th for exit)
    bb_width_pct = np.full(n, np.nan)
    for i in range(20, n):
        window = bb_width[max(0, i-50):i+1]  # 50-bar lookback for percentile
        if len(window) > 0 and not np.all(np.isnan(window)):
            valid_widths = window[~np.isnan(window)]
            if len(valid_widths) >= 10:
                bb_width_pct[i] = (np.sum(valid_widths <= bb_width[i]) / len(valid_widths)) * 100
    
    # Volume confirmation on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    if len(vol_12h) >= 20:
        vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_spike_12h = vol_12h > (2.0 * vol_ma_20_12h)
    else:
        vol_spike_12h = np.zeros(len(df_12h), dtype=bool)
    
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # Camarilla pivot levels on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 1:  # need previous day
            pd_high = high_1d[i-1]
            pd_low = low_1d[i-1]
            pd_close = close_1d[i-1]
            diff = pd_high - pd_low
            camarilla_r3[i] = pd_close + (1.1 * diff / 2)  # R3 = C + 1.1*(H-L)/2
            camarilla_s3[i] = pd_close - (1.1 * diff / 2)  # S3 = C - 1.1*(H-L)/2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_width_pct[i]) or np.isnan(bb_ma[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze (width < 20th percentile) AND volume spike AND price breaks above R3
            if (bb_width_pct[i] < 20.0 and 
                vol_spike_12h_aligned[i] == 1.0 and 
                close[i] > camarilla_r3_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze (width < 20th percentile) AND volume spike AND price breaks below S3
            elif (bb_width_pct[i] < 20.0 and 
                  vol_spike_12h_aligned[i] == 1.0 and 
                  close[i] < camarilla_s3_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to BB middle OR BB width expands (>50th percentile)
            if (close[i] <= bb_ma[i] or bb_width_pct[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to BB middle OR BB width expands (>50th percentile)
            if (close[i] >= bb_ma[i] or bb_width_pct[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals