#!/usr/bin/env python3
"""
1D_TRIX_VolumeSpike_ChopRegime
Hypothesis: Daily TRIX (triple-smoothed EMA) with volume spike and chop regime filter.
TRIX captures momentum reversals with less lag than MACD. Volume spike confirms breakout strength.
Chop regime filter avoids whipsaws in sideways markets. Works in bull/bear by capturing momentum shifts.
Targets 7-25 trades/year on 1d timeframe to minimize fee drag.
"""
name = "1D_TRIX_VolumeSpike_ChopRegime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100  # Percentage change
    trix = trix.fillna(0).values
    
    # Calculate 1W EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current daily volume > 2.0 x 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    # Chop regime filter: Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
    # Chop = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high - min_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Handle division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(45, 20, 14)  # Ensure sufficient warmup for TRIX and other indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 15 days between trades to reduce frequency
            if bars_since_exit < 15:
                continue
                
            # Long: TRIX turns positive with volume spike and chop < 61.8 (not too choppy)
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_filter[i] and chop[i] < 61.8):
                # Only take long if 1W trend is up (EMA34 rising)
                if i > 0 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
            # Short: TRIX turns negative with volume spike and chop < 61.8 (not too choppy)
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_filter[i] and chop[i] < 61.8):
                # Only take short if 1W trend is down (EMA34 falling)
                if i > 0 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    bars_since_exit = 0
        elif position != 0:
            # Exit: TRIX crosses zero (momentum reversal)
            if position == 1 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals