#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels.
# Uses 1w EMA(34) for trend filter to avoid counter-trend trades.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe.
# Works in bull/bear: 1w EMA ensures trend alignment, Camarilla provides structured levels.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels ===
    # Calculate from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    r4 = prev_close + (prev_range * 1.1 / 2)
    s4 = prev_close - (prev_range * 1.1 / 2)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(34) for long-term trend bias
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.3x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Breakout above R4 with volume and trend alignment
        # 2. Mean reversion bounce from S3 with volume and trend alignment
        if vol_confirm:
            if ((close[i] > r4_aligned[i] and close[i] > ema_34_1w_aligned[i]) or
                (close[i] > s3_aligned[i] and close[i] < ema_34_1w_aligned[i])):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Breakdown below S4 with volume and trend alignment
        # 2. Mean reversion rejection at R3 with volume and trend alignment
        elif vol_confirm:
            if ((close[i] < s4_aligned[i] and close[i] < ema_34_1w_aligned[i]) or
                (close[i] < r3_aligned[i] and close[i] > ema_34_1w_aligned[i])):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_VolTrendFilter_v1"
timeframe = "6h"
leverage = 1.0