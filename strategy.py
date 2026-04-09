#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1d EMA200 trend + volume spike confirmation
# Camarilla levels from 1d provide institutional support/resistance; breakout at R4/S4 with volume confirms institutional participation
# 1d EMA200 filter ensures alignment with major trend to avoid counter-trend whipsaws
# Works in bull/bear: trend filter adapts, Camarilla breakouts capture acceleration moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_camarilla_ema200_volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # Breakout level for longs
    camarilla_s4 = np.full(n, np.nan)  # Breakout level for shorts
    camarilla_r3 = np.full(n, np.nan)  # Fade level for longs
    camarilla_s3 = np.full(n, np.nan)  # Fade level for shorts
    camarilla_pivot = np.full(n, np.nan)
    
    # Calculate Camarilla for each 1d bar using previous day's OHLC
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day
            continue
        phigh = df_1d['high'].iloc[i-1]
        plow = df_1d['low'].iloc[i-1]
        pclose = df_1d['close'].iloc[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_ = phigh - plow
        
        camarilla_pivot.iloc[i] = pivot
        camarilla_r4.iloc[i] = pivot + range_ * 1.1/2
        camarilla_s4.iloc[i] = pivot - range_ * 1.1/2
        camarilla_r3.iloc[i] = pivot + range_ * 1.1/4
        camarilla_s3.iloc[i] = pivot - range_ * 1.1/4
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6h average volume for volume spike confirmation (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla S3 (mean reversion) OR price < EMA200 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 (mean reversion) OR price > EMA200 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume spike and Camarilla breakout + EMA200 filter
            if volume_spike:
                # Long entry: price > Camarilla R4 AND price > EMA200 (bullish breakout with trend)
                if close[i] > camarilla_r4_aligned[i] and close[i] > ema_200_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla S4 AND price < EMA200 (bearish breakout with trend)
                elif close[i] < camarilla_s4_aligned[i] and close[i] < ema_200_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals