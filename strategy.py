#!/usr/bin/env python3
# 4h_1d_Momentum_Pullback_Volume_TrendFilter
# Hypothesis: Buy pullbacks to the 4h EMA20 in strong trends (EMA20 > EMA50) with volume confirmation and 1d trend filter; sell rallies to EMA20 in downtrends. Uses 1d EMA50 to filter counter-trend trades. Designed for low trade frequency (~25/year) to avoid fee drag, works in bull/bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Momentum_Pullback_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4h: EMA20 and EMA50 for momentum and dynamic support/resistance ===
    close = prices['close'].values
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = close[i]
        ema20_val = ema20[i]
        ema50_val = ema50[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(ema50_val) or np.isnan(ema50_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price pulls back to EMA20 in uptrend (EMA20 > EMA50 and 1d EMA50 rising)
            if (close_val <= ema20_val * 1.005 and  # Allow small tolerance for pullback
                ema20_val > ema50_val and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] if i > 0 else False and  # 1d EMA50 rising
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Price rallies to EMA20 in downtrend (EMA20 < EMA50 and 1d EMA50 falling)
            elif (close_val >= ema20_val * 0.995 and  # Allow small tolerance for rally
                  ema20_val < ema50_val and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] if i > 0 else False and  # 1d EMA50 falling
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA50 or loses momentum
            if close_val < ema50_val or ema20_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA50 or loses momentum
            if close_val > ema50_val or ema20_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals