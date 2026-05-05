#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 1d ATR ratio < 0.8 (low volatility regime) AND volume spike
# Short when price breaks below 20-period Donchian low AND 1d ATR ratio < 0.8 AND volume spike
# Donchian channels provide clear structural breaks with good risk/reward
# 1d ATR ratio (current ATR/20-period MA ATR) filters for low volatility environments where breakouts are more reliable
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity with fee drag
# Works in bull (trend continuation breaks) and bear (mean reversion fails, breakouts capture panic moves)
# Timeframe: 4h (primary timeframe as required)

name = "4h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d ATR and its 20-period MA for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    
    # 1d ATR (20-period)
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    # 20-period MA of ATR
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    # ATR ratio: current ATR / MA of ATR (< 0.8 = low volatility regime)
    atr_ratio = np.where(atr_ma_20 > 0, atr_1d / atr_ma_20, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    # Donchian High: 20-period rolling max of high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian Low: 20-period rolling min of low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND low volatility regime AND volume spike
            if (close[i] > donch_high[i] and 
                atr_ratio_aligned[i] < 0.8 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND low volatility regime AND volume spike
            elif (close[i] < donch_low[i] and 
                  atr_ratio_aligned[i] < 0.8 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low (mean reversion) OR high volatility regime
            if close[i] < donch_low[i] or atr_ratio_aligned[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high OR high volatility regime
            if close[i] > donch_high[i] or atr_ratio_aligned[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals