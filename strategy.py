#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Reversal with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# 1w EMA34 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.5 x 20-period EMA) confirms reversal validity
# Long: %R crosses above -80 (oversold) in uptrend with volume confirmation
# Short: %R crosses below -20 (overbought) in downtrend with volume confirmation
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_WilliamsR_Reversal_1wEMA34_Trend_VolumeConfirmation"
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
    
    # 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 calculation
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R and EMA calculation)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Williams %R signals: cross above -80 (long), cross below -20 (short)
        williams_r_long = williams_r[i] > -80 and (i == start_idx or williams_r[i-1] <= -80)
        williams_r_short = williams_r[i] < -20 and (i == start_idx or williams_r[i-1] >= -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 with volume confirmation and uptrend
            if williams_r_long and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 with volume confirmation and downtrend
            elif williams_r_short and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend changes to downtrend
            if williams_r[i] < -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend changes to uptrend
            if williams_r[i] > -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals