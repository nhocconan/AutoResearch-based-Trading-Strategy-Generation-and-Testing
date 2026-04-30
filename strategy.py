#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike
# Uses 6h primary timeframe to target 50-150 trades over 4 years (12-37/year).
# Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
# Entries: long when %R crosses above -80 from below, short when %R crosses below -20 from above.
# Trend filter: 1d EMA34 slope > 0 for longs, < 0 for shorts ensures trading with higher timeframe trend.
# Volume confirmation: 6h volume > 1.5x 20-period average reduces false signals.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Works in bull via dip buying,
# in bear via rally selling with trend alignment.

name = "6h_WilliamsR_ME_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 6h Williams %R(14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Calculate EMA slope (change over 3 periods)
    ema_slope = np.zeros_like(ema_34_aligned)
    ema_slope[3:] = ema_34_aligned[3:] - ema_34_aligned[:-3]
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_slope[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1]
        curr_ema_slope = ema_slope[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Long: Williams %R crosses above -80 from below AND EMA slope up
                if prev_williams_r <= -80 and curr_williams_r > -80 and curr_ema_slope > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND EMA slope down
                elif prev_williams_r >= -20 and curr_williams_r < -20 and curr_ema_slope < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought)
            if curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold)
            if curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals