#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RangeReversion_Volume_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for volatility-based range
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w range-based bands from previous week
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    prev_range = high_1w[:-1] - low_1w[:-1]
    prev_range = np.concatenate([[np.nan], prev_range])
    
    # Use adaptive multiplier based on ATR percentile for regime adaptation
    atr_series = pd.Series(atr_10)
    atr_percentile = atr_series.rolling(window=30, min_periods=10).rank(pct=True).values
    # Volatility multiplier: 0.5 in low vol, 1.5 in high vol
    vol_multiplier = 0.5 + atr_percentile
    
    # Mean reversion bands: tighter in high vol, wider in low vol
    multiplier = 0.8 * vol_multiplier  # Base multiplier adjusted by volatility
    
    upper_band = prev_close + prev_range * multiplier
    lower_band = prev_close - prev_range * multiplier
    
    # Align bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price touches or breaks below lower band with volume (mean reversion long)
            if price <= lower_band_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks above upper band with volume (mean reversion short)
            elif price >= upper_band_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midline or reverse signal
            midline = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if price >= midline:
                signals[i] = 0.0
                position = 0
            elif price >= upper_band_aligned[i] and volume_ok:
                # Reverse to short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midline or reverse signal
            midline = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if price <= midline:
                signals[i] = 0.0
                position = 0
            elif price <= lower_band_aligned[i] and volume_ok:
                # Reverse to long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals