#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA50 trend filter + volume spike.
# Williams %R(14) = (Highest High(14) - Close) / (Highest High(14) - Lowest Low(14)) * -100
# Long when Williams %R crosses above -80 from below (oversold reversal), price > 1d EMA50, volume > 2x average.
# Short when Williams %R crosses below -20 from above (overbought reversal), price < 1d EMA50, volume > 2x average.
# Williams %R identifies momentum reversals; 1d EMA50 filters for higher timeframe trend alignment.
# Volume spike confirms institutional participation in the reversal. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Detect crosses: above -80 (long signal) or below -20 (short signal)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    cross_above_80 = (williams_r > -80) & (williams_r_prev <= -80)
    cross_below_20 = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Williams %R crosses above -80, price above 1d EMA50, volume filter
        if cross_above_80[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short condition: Williams %R crosses below -20, price below 1d EMA50, volume filter
        elif cross_below_20[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0