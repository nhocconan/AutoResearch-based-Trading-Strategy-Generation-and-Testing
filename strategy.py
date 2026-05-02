#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Reversal with 1d EMA34 trend filter and volume spike
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# provide high-probability mean reversion entries in ranging markets.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades in strong trends.
# Volume confirmation filters low-momentum false signals.
# Works in bull markets (oversold reversals during uptrend) and bear markets 
# (overbought reversals during downtrend). Discrete position sizing (0.25) controls risk.

name = "6h_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R calculation (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R calculation)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) with volume confirmation and uptrend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume confirmation and downtrend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend changes to downtrend
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend changes to uptrend
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals