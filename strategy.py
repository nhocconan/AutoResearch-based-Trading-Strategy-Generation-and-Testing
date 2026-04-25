#!/usr/bin/env python3
"""
1d_WilliamsVIXFix_MeanReversion_BearRegime
Hypothesis: In bear markets (2025+), BTC/ETH exhibit mean-reverting spikes. Williams VIX Fix identifies volatility spikes (like VIX) that precede reversals. Long when VIX Fix > 80 (extreme fear) and price below 200 EMA (bearish context). Short when VIX Fix < 20 (extreme complacency) and price above 200 EMA. Use 1d timeframe for low trade frequency (<25/year) to minimize fee drag. 1w EMA50 trend filter ensures alignment with higher timeframe direction. Volume spike (>2.0x 20-bar MA) confirms exhaustion. Discrete sizing 0.25 balances risk and return.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams VIX Fix calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams VIX Fix: measures market fear/volatility
    # Formula: ((Highest Close in Period - Low) / (Highest Close in Period - Lowest Close in Period)) * 100
    # We use 22-day period (approx 1 month) similar to VIX calculation
    highest_close_22 = pd.Series(close_1d).rolling(window=22, min_periods=22).max().values
    lowest_close_22 = pd.Series(close_1d).rolling(window=22, min_periods=22).min().values
    
    # Avoid division by zero
    denominator = highest_close_22 - lowest_close_22
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    vix_fix = ((highest_close_22 - low_1d) / denominator) * 100
    vix_fix_aligned = align_htf_to_ltf(prices, df_1d, vix_fix)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 200 EMA on 1d for bear/bull context
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for longest indicator (200 EMA)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vix_fix_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Extreme fear (VIX Fix > 80) AND bearish context (price < 200 EMA) AND volume spike
            long_setup = (vix_fix_aligned[i] > 80) and \
                         (close[i] < ema_200_1d_aligned[i]) and \
                         volume_confirm[i]
            # Short: Extreme complacency (VIX Fix < 20) AND bullish context (price > 200 EMA) AND volume spike
            short_setup = (vix_fix_aligned[i] < 20) and \
                          (close[i] > ema_200_1d_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: VIX Fix normalizes (< 50) OR price reaches 200 EMA (mean reversion complete)
            if (vix_fix_aligned[i] < 50) or (close[i] >= ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: VIX Fix normalizes (> 50) OR price reaches 200 EMA (mean reversion complete)
            if (vix_fix_aligned[i] > 50) or (close[i] <= ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsVIXFix_MeanReversion_BearRegime"
timeframe = "1d"
leverage = 1.0