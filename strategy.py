#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 55-period Donchian breakout with 1w ATR filter and volume confirmation.
# Works in bull (breakouts continue) and bear (mean reversion at opposite channel) via volatility filtering.
# Target: 7-25 trades/year (30-100 total over 4 years) to avoid fee drag.
name = "1d_Donchian55_1wATR_VolumeFilter_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 55-period Donchian channels (upper/lower)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=55, min_periods=55).max().values
    donchian_lower = low_series.rolling(window=55, min_periods=55).min().values
    
    # Calculate weekly ATR (14) for volatility filter
    tr1 = np.abs(df_1w['high'] - df_1w['low'])
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to daily (wait for weekly close)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Volume filter: current volume > 2.0 * 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume and volatility filter
            if close_val > upper and vol_filter and (atr_val > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and volatility filter
            elif close_val < lower and vol_filter and (atr_val > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower Donchian or volatility drops
            if close_val < lower or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Donchian or volatility drops
            if close_val > upper or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals