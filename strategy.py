#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold), price > 1w EMA50, and volume > 2.0x 20-bar avg.
# Short when Williams %R > -20 (overbought), price < 1w EMA50, and volume > 2.0x 20-bar avg.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Uses 1w EMA50 for strong higher timeframe trend filter to avoid counter-trend trades.
# Williams %R(14) identifies extreme momentum exhaustion points.
# Volume confirmation with 2.0x threshold reduces false signals.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets: trend filter ensures we trade with the 1w trend,
# while Williams %R provides mean reversion entries during pullbacks.

name = "6h_WilliamsR_MeanRev_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Williams %R(14) and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), price > 1w EMA50, volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < 1w EMA50, volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses back above -50 (momentum returning)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses back below -50 (momentum returning)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals