#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels (>80 or <20) 
# provide high-probability mean-reversion entries in ranging markets. 
# 1d EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Volume spike (>2.0 x 30-period EMA) confirms momentum behind the reversal.
# Discrete position sizing (0.25) balances opportunity with fee drag control.
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns.
# Works in both bull and bear markets by fading extremes in ranging regimes while respecting trend.

name = "12h_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) with volume confirmation and not strong downtrend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmation[i] and not downtrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume confirmation and not strong uptrend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmation[i] and not uptrend:
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