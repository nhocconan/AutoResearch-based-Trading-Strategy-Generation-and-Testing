#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R mean reversion with volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; mean reversion works in both bull and bear markets.
# Long when Williams %R < -80 (oversold) AND volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought) AND volume > 1.5x 20-bar average.
# Exit when Williams %R crosses -50 midpoint.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsR_MeanRev_1d_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period Williams %R
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        williams_r_val = williams_r_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold (Williams %R < -80) AND volume confirmation
            if williams_r_val < -80 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: overbought (Williams %R > -20) AND volume confirmation
            elif williams_r_val > -20 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses above -50 (mean reversion complete)
            if williams_r_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses below -50 (mean reversion complete)
            if williams_r_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals