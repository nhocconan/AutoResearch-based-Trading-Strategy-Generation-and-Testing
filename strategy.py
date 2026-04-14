#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and ATR volatility filter.
# Long when price breaks above 1d Donchian high with volume > 1.5x 20-period average and ATR(14) > median ATR.
# Short when price breaks below 1d Donchian low with volume > 1.5x 20-period average and ATR(14) > median ATR.
# Exit when price returns to 1d close or ATR falls below median (volatility collapse).
# Designed to capture high-momentum breakouts in volatile markets while avoiding low-volatility false breakouts.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Donchian channels (20-period)
    lookback = 20
    donch_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Median ATR for volatility filter
    atr_median = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    
    # Prior 1d close for exit condition
    prior_close_1d = np.roll(close_1d, 1)
    prior_close_1d[0] = np.nan
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    prior_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(lookback, atr_period, 50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_median_aligned[i]) or
            np.isnan(prior_close_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume and volatility confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        volatility_confirmed = atr_1d_aligned[i] > atr_median_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND volume/volatility confirmed
            if (close[i] > donch_high_aligned[i] and 
                volume_confirmed and 
                volatility_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND volume/volatility confirmed
            elif (close[i] < donch_low_aligned[i] and 
                  volume_confirmed and 
                  volatility_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to prior 1d close or volatility collapses
            if (close[i] <= prior_close_1d_aligned[i] or 
                not volatility_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to prior 1d close or volatility collapses
            if (close[i] >= prior_close_1d_aligned[i] or 
                not volatility_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0