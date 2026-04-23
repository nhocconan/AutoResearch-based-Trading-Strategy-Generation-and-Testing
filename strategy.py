#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1d Trend Filter and Volume Confirmation
- Williams %R measures overbought/oversold conditions (-20 to 0 = overbought, -80 to -100 = oversold)
- Long when %R crosses above -80 from below (oversold bounce) with 1d uptrend and volume confirmation
- Short when %R crosses below -20 from above (overbought rejection) with 1d downtrend and volume confirmation
- 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via 1d trend filter and volatility-adjusted signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 30, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) with 1d uptrend and volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1d_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) with 1d downtrend and volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1d_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or trend breaks
            if williams_r[i] >= -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or trend breaks
            if williams_r[i] <= -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0