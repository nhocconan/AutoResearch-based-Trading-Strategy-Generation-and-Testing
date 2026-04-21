#!/usr/bin/env python3
"""
1d_Range_Mean_Reversion_With_Weekly_Trend_Filter
Hypothesis: Use daily mean reversion at Bollinger Bands (20, 2) with weekly trend filter (EMA 50) and volume confirmation.
Long when price touches lower BB with volume spike > 1.5 and above weekly EMA50.
Short when price touches upper BB with volume spike > 1.5 and below weekly EMA50.
Exit when price crosses 20-day SMA. Designed to capture mean reversion in ranging markets while avoiding counter-trend trades.
Target ~10-20 trades/year per symbol by requiring multiple confluence conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Bollinger Bands (20, 2) ===
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_bb = sma_20 - 2 * std_20
    upper_bb = sma_20 + 2 * std_20
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(lower_bb[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1w = ema_50_1w_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price touches lower BB + volume spike > 1.5 + above weekly EMA50
            if (price_close <= lower_bb[i] and 
                vol_spike > 1.5 and 
                price_close > trend_1w):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper BB + volume spike > 1.5 + below weekly EMA50
            elif (price_close >= upper_bb[i] and 
                  vol_spike > 1.5 and 
                  price_close < trend_1w):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses 20-day SMA (mean reversion complete)
            if position == 1 and price_close >= sma_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close <= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Range_Mean_Reversion_With_Weekly_Trend_Filter"
timeframe = "1d"
leverage = 1.0