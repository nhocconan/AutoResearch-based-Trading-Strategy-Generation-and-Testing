#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) reversal with 12h EMA(34) trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets, reversals from extreme
# levels often signal continuation rather than reversal. We use 12h EMA to determine trend direction
# and only take trades in the direction of the trend: buy when %R crosses above -80 in uptrend,
# sell when %R crosses below -20 in downtrend. Volume spike confirms momentum.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via trend filter - only trades with the trend, avoiding counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams %R and trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) for each 12h bar
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe (waits for 12h bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold recovery) + uptrend + volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
                close[i] > ema_34_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) + downtrend + volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on overbought condition or trend reversal
                if (williams_r_aligned[i] < -20 or close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on oversold condition or trend reversal
                if (williams_r_aligned[i] > -80 or close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0