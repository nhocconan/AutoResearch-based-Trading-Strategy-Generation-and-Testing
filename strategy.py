#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot breakout (R3/S3) with volume confirmation and 1w EMA50 trend filter.
# Uses 1d Camarilla R3/S3 levels for entry, filtered by 1w EMA50 trend and volume spike.
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull/bear:
# - 1w EMA50 avoids counter-trend trades in strong trends
# - Camarilla R3/S3 breakouts capture strong momentum after range expansion
# - Volume confirmation ensures institutional participation
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivots (R3, S3) ===
    # Camarilla formula: Close +- (High-Low) * 1.1/4
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 4  # R3 level
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 4   # S3 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(50) for trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d strong resistance)
        # 2. 1w price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > camarilla_high_aligned[i] and
            close[i] > ema_50_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d strong support)
        # 2. 1w price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < camarilla_low_aligned[i] and
              close[i] < ema_50_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_EMA50_1w_VolFilter_v1"
timeframe = "12h"
leverage = 1.0