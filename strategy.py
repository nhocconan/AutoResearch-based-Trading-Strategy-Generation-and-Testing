#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In trending markets (1d EMA200),
# we take counter-trend reversals at extremes with volume confirmation.
# Works in bull/bear by fading extremes in the direction of higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_WilliamsR_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h Williams %R (14-period) ===
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full_like(high_4h, np.nan)
    lowest_low = np.full_like(low_4h, np.nan)
    
    for i in range(13, len(high_4h)):
        highest_high[i] = np.max(high_4h[i-13:i+1])
        lowest_low[i] = np.min(low_4h[i-13:i+1])
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full_like(close_4h, np.nan)
    for i in range(13, len(high_4h)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_4h[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # === 4h Volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Get values
        close_val = close_4h[i]
        wr_val = williams_r[i]
        ema200_val = ema200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(wr_val) or np.isnan(ema200_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above 1d EMA200 (uptrend) AND volume confirmation
            if wr_val < -80 and close_val > ema200_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below 1d EMA200 (downtrend) AND volume confirmation
            elif wr_val > -20 and close_val < ema200_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) OR trend breaks
            if wr_val > -50 or close_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) OR trend breaks
            if wr_val < -50 or close_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals