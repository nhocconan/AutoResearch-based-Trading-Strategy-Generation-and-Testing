#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ATR regime filter.
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-bar average AND 1d ATR(14) < 0.03*close (low volatility regime).
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-bar average AND 1d ATR(14) < 0.03*close.
# Exit when price crosses the Donchian midpoint (mean of upper and lower bands).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 75-150 total trades over 4 years (19-38/year). Works in bull/bear via volatility regime filter.

name = "4h_Donchian20_VolumeSpike_LowVol_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize ATR by 1d close to get percentage
    atr_pct_1d = atr_14_1d / close_1d
    atr_pct_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    
    # Calculate Donchian(20) channels on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(atr_pct_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Low volatility regime filter: ATR% < 3%
        low_vol_regime = atr_pct_1d_aligned[i] < 0.03
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper band, low volatility regime, volume confirmation
            if (curr_high > highest_20[i] and 
                low_vol_regime and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, low volatility regime, volume confirmation
            elif (curr_low < lowest_20[i] and 
                  low_vol_regime and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian midpoint
            if curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian midpoint
            if curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals