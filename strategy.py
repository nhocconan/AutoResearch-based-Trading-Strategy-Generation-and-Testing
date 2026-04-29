#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + 1w EMA34 Trend Filter
# Long when Williams %R < -80 (oversold), 1d volume > 1.5x 20-period average, and close > 1w EMA34
# Short when Williams %R > -20 (overbought), 1d volume > 1.5x 20-period average, and close < 1w EMA34
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete position sizing (0.25) to limit fee drag and manage drawdown.
# Williams %R identifies extreme price reversals, volume spike confirms participation,
# 1w EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to stay within trade limits.
# Works in both bull and bear markets by fading extremes only when volume confirms and trend aligns.

name = "4h_WilliamsR_VolumeSpike_1wEMA34_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period)
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 34)  # Williams %R, volume MA, and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_williams_r = williams_r[i]
        curr_vol_ma_20 = vol_ma_20_aligned[i]
        curr_volume = volume[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_close = close[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = curr_volume > 1.5 * curr_vol_ma_20
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (exiting oversold)
            if curr_williams_r > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (exiting overbought)
            if curr_williams_r < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold), volume spike, and close > 1w EMA34
            if curr_williams_r < -80.0 and volume_spike and curr_close > curr_ema34_1w:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought), volume spike, and close < 1w EMA34
            elif curr_williams_r > -20.0 and volume_spike and curr_close < curr_ema34_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals