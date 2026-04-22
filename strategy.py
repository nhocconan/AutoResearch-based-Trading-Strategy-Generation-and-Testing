#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily timeframe strategy
    # Hypothesis: Combine weekly Bollinger Band squeeze with daily price action and volume
    # Works in bull markets (breakouts from squeeze) and bear markets (mean reversion at bands)
    # Low frequency trading to minimize fee drag
    
    # Load weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # Align weekly Bollinger Bands to daily
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # Daily ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(sma_20_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze condition (low volatility)
            squeeze = bb_width_aligned[i] < 0.03  # 3% width threshold
            
            # Long: Price breaks above upper band with volume surge during squeeze
            if squeeze and close[i] > upper_band_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with volume surge during squeeze
            elif squeeze and close[i] < lower_band_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle band (mean reversion)
            if position == 1:
                if close[i] < sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_BollingerSqueeze_Breakout_WeeklyBB_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0