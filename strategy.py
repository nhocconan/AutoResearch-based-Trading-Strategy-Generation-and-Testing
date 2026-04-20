#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20-period) with 1-day volume confirmation and volatility filter
# Long: Price breaks above upper band AND volatility low (avoid whipsaw) AND volume spike
# Short: Price breaks below lower band AND volatility low AND volume spike
# Exit: Opposite band touch or volatility expansion (whipsaw protection)
# Uses 1-day volatility filter (ATR ratio) to avoid false breakouts in high volatility
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-day ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30d = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    # Volatility filter: low volatility environment (short ATR < long ATR * 0.8)
    vol_filter = atr_10d < (atr_30d * 0.8)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate 20-period Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper and lower bands
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day volume spike confirmation
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(vol_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        upper_band = upper[i]
        lower_band = lower[i]
        vol_ok = vol_filter_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper band, low volatility, volume spike
            if price > upper_band and vol_ok and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band, low volatility, volume spike
            elif price < lower_band and vol_ok and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: touch lower band OR volatility expansion (whipsaw protection)
            if price < lower_band or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch upper band OR volatility expansion
            if price > upper_band or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0