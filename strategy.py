#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R crosses above -80 from below (oversold reversal) in uptrend (price > 1d EMA34)
# Short when %R crosses below -20 from above (overbought reversal) in downtrend (price < 1d EMA34)
# Volume confirmation (>1.5 x 20-period EMA) ensures breakout validity
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought rejections in downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

name = "4h_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirm"
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
    
    # 14-period Williams %R calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R calculation)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R reversal signals
        williams_r_long = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80
        williams_r_short = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold reversal with volume confirmation and uptrend
            if williams_r_long and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought reversal with volume confirmation and downtrend
            elif williams_r_short and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R becomes overbought (> -20) OR trend changes to downtrend
            if williams_r[i] >= -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R becomes oversold (< -80) OR trend changes to uptrend
            if williams_r[i] <= -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals