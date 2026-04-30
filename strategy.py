#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + 12h EMA50 trend filter + volume spike confirmation.
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Long when Williams %R < -80 (oversold) AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Short when Williams %R > -20 (overbought) AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn. Works in bull/bear via 12h EMA50 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsR_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # warmup for EMA50, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), uptrend, volume spike
            if (curr_williams_r < -80 and is_uptrend and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), downtrend, volume spike
            elif (curr_williams_r > -20 and is_downtrend and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R >= -20 (overbought) or trend changes
            if curr_williams_r >= -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R <= -80 (oversold) or trend changes
            if curr_williams_r <= -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals