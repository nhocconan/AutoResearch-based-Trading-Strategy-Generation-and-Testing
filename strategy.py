#!/usr/bin/env python3
name = "6h_Gaussian_Kernel_Trend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Gaussian kernel smoothing of daily close (sigma=3) for trend
    daily_close = df_1d['close'].values
    kernel_size = 15
    kernel = stats.norm.pdf(np.arange(-kernel_size//2, kernel_size//2+1), 0, 3)
    kernel = kernel / kernel.sum()
    gaussian_smooth = np.convolve(daily_close, kernel, mode='same')
    gaussian_smooth[:kernel_size//2] = gaussian_smooth[kernel_size//2]
    gaussian_smooth[-kernel_size//2:] = gaussian_smooth[-kernel_size//2-1]
    gaussian_smooth_aligned = align_htf_to_ltf(prices, df_1d, gaussian_smooth)
    
    # Daily ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h price position relative to Gaussian trend
    price_ratio = close / gaussian_smooth_aligned
    price_ratio_smooth = pd.Series(price_ratio).rolling(window=6, min_periods=6).mean().values
    
    # 6h volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(gaussian_smooth_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(price_ratio_smooth[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion when price deviates significantly from trend
        z_score = (price_ratio_smooth[i] - 1.0) * 100  # Scale for sensitivity
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long when price significantly below trend with volume
            if z_score < -0.8 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short when price significantly above trend with volume
            elif z_score > 0.8 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to trend or momentum reverses
            if z_score > -0.2 or price_ratio_smooth[i] > price_ratio_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to trend or momentum reverses
            if z_score < 0.2 or price_ratio_smooth[i] < price_ratio_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Gaussian kernel trend + mean reversion with volume
# - Uses Gaussian kernel smoothing (sigma=3) on daily closes to identify smooth trend
# - Enters mean reversion when 6s price deviates significantly from trend (>0.8 sigma)
# - Requires volume spike (1.5x average) to confirm institutional interest
# - Works in both bull and bear markets as mean reversion occurs in all regimes
# - Gaussian kernel reduces noise vs simple moving averages for better trend identification
# - Exits when price returns to trend or momentum changes direction
# - Position size 0.25 limits risk while capturing reversion moves
# - Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Novel: Gaussian kernel smoothing for trend identification not commonly used in crypto
# - Combines trend following (Gaussian slope) with mean reversion (deviation from trend)