# Based on experiment #68405: 12h timeframe with HTF=1d
# Hypothesis: 12h Donchian breakout with daily volume and ATR filter
# - Uses 12h Donchian(20) for breakout signals
# - Filters with daily volume > 1.5x 20-day average (volume confirmation)
# - Uses daily ATR for volatility filter (ATR > 0.5 * 20-day ATR average)
# - Only takes long when price breaks above upper band, short when breaks below lower band
# - Position size: 0.25 (25% of capital) to manage drawdown
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 12h timeframe
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_ma_20_12h = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20_12h[i]) or np.isnan(atr_ma_20_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume and volatility filters
        volume_filter = volume_1d[i] > 1.5 * vol_ma_20_12h[i] if not np.isnan(volume_1d[i]) else False
        vol_filter = atr_14[i] > 0.5 * atr_ma_20_12h[i] if not np.isnan(atr_14[i]) else False
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band + filters
            if close_12h[i] > donchian_high[i] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian band + filters
            elif close_12h[i] < donchian_low[i] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band
            if close_12h[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band
            if close_12h[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_VolumeATRFilter"
timeframe = "12h"
leverage = 1.0