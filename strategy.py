#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d EMA34 trend filter and volume spike confirmation.
- Williams %R (14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- Long signal: Williams %R crosses above -80 from below (reversal from oversold) AND price > 1d EMA34 (uptrend filter) AND volume > 1.8x 20-bar average.
- Short signal: Williams %R crosses below -20 from above (reversal from overbought) AND price < 1d EMA34 (downtrend filter) AND volume > 1.8x 20-bar average.
- Uses 6h timeframe to capture swing reversals with lower frequency than lower timeframes.
- Discrete position size 0.25 to manage drawdown and reduce fee churn.
- Targets 50-150 total trades over 4 years (12-37/year) to stay fee-efficient.
- Volume confirmation reduces false reversals in choppy markets.
- Williams %R is effective in ranging and trending markets, working in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough for EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Williams %R reversal signals
        williams_r_long_signal = (williams_r[i-1] <= -80) and (williams_r[i] > -80)  # Cross above -80
        williams_r_short_signal = (williams_r[i-1] >= -20) and (williams_r[i] < -20)  # Cross below -20
        
        if position == 0:
            # Only trade if volume confirms reversal
            if volume_confirm:
                # Long: Williams %R reverses above -80 AND price above 1d EMA34 (uptrend)
                if williams_r_long_signal and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R reverses below -20 AND price below 1d EMA34 (downtrend)
                elif williams_r_short_signal and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price crosses below 1d EMA34
            if (williams_r[i] >= -20) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price crosses above 1d EMA34
            if (williams_r[i] <= -80) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0