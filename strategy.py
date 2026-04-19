#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ThreeTierBreakout_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility-based bands
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for adaptive band width
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).rank(pct=True).values
    # Volatility multiplier: 0.5 to 1.5
    vol_multiplier = 0.5 + atr_percentile
    
    # Calculate 1d range-based bands from previous day
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_range = high_1d[:-1] - low_1d[:-1]
    prev_range = np.concatenate([[np.nan], prev_range])
    
    # Three-tier system: tight, medium, wide bands
    tight_multiplier = 0.3 * vol_multiplier
    medium_multiplier = 0.6 * vol_multiplier
    wide_multiplier = 1.0 * vol_multiplier
    
    tight_upper = prev_close + prev_range * tight_multiplier
    tight_lower = prev_close - prev_range * tight_multiplier
    medium_upper = prev_close + prev_range * medium_multiplier
    medium_lower = prev_close - prev_range * medium_multiplier
    wide_upper = prev_close + prev_range * wide_multiplier
    wide_lower = prev_close - prev_range * wide_multiplier
    
    # Align all bands to 4h timeframe
    tight_upper_aligned = align_htf_to_ltf(prices, df_1d, tight_upper)
    tight_lower_aligned = align_htf_to_ltf(prices, df_1d, tight_lower)
    medium_upper_aligned = align_htf_to_ltf(prices, df_1d, medium_upper)
    medium_lower_aligned = align_htf_to_ltf(prices, df_1d, medium_lower)
    wide_upper_aligned = align_htf_to_ltf(prices, df_1d, wide_upper)
    wide_lower_aligned = align_htf_to_ltf(prices, df_1d, wide_lower)
    
    # Volume filter: current volume > 1.5x 24-period average (24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)
    
    for i in range(start_idx, n):
        if np.isnan(tight_upper_aligned[i]) or np.isnan(tight_lower_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above medium band with volume
            if price > medium_upper_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below medium band with volume
            elif price < medium_lower_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: return to tight band or reverse signal
            if price < tight_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < medium_lower_aligned[i] and volume_ok:
                # Reverse to short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: return to tight band or reverse signal
            if price > tight_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > medium_upper_aligned[i] and volume_ok:
                # Reverse to long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals