#!/usr/bin/env python3
"""
1-day Williams %R mean reversion with 1-week trend filter and volume confirmation
Hypothesis: Williams %R extreme readings (-80/-20) signal mean reversion opportunities
when aligned with the weekly trend direction, confirmed by volume spikes. The weekly
trend filter prevents counter-trend trades in strong trends, while Williams %R
captures reversals in both bull and bear markets. Designed for low trade frequency
(10-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_williams_r_mean_reversion_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams %R (14-period) - measures overbought/oversold
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend turns bearish
            if (williams_r[i] >= -20 or 
                close[i] < ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend turns bullish
            if (williams_r[i] <= -80 or 
                close[i] > ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Williams %R below -80 (oversold) + volume spike + weekly uptrend
            if (williams_r[i] < -80 and
                vol_spike[i] and
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R above -20 (overbought) + volume spike + weekly downtrend
            elif (williams_r[i] > -20 and
                  vol_spike[i] and
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals