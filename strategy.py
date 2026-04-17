#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-bar average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-bar average.
Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is hit.
Uses 1d for EMA trend regime and 6h for execution, Williams %R, and volume.
Designed to capture mean reversion in ranging markets and pullbacks in trending markets across bull and bear.
Target: 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 6h volume MA (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (6h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        mean_revert = abs(williams_r_aligned[i] + 50) < 5  # near -50
        
        # Exit conditions: mean reversion or opposite extreme
        exit_long = mean_revert or williams_r_aligned[i] > -20
        exit_short = mean_revert or williams_r_aligned[i] < -80
        
        if position == 0:
            # Long: oversold + uptrend + volume confirmation
            if (oversold and above_ema and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume confirmation
            elif (overbought and below_ema and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: mean reversion or overbought
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: mean reversion or oversold
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0