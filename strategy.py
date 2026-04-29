#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: Close > Donchian High(20) AND price > 1d EMA34 AND volume > 2.0x 20-bar average
# Short: Close < Donchian Low(20) AND price < 1d EMA34 AND volume > 2.0x 20-bar average
# Exit: Close crosses below/above 1d EMA34 OR Donchian middle line
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
# Works in bull via breakout continuation, in bear via mean reversion at extremes with trend filter.

name = "12h_DonchianBreakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_rolling_max + low_rolling_min) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # max(20, 34) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = high_rolling_max[i]
        curr_donchian_low = low_rolling_min[i]
        curr_donchian_mid = donchian_mid[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Close below 1d EMA34 (trend change) OR close below Donchian middle (mean reversion)
            if curr_close < curr_ema_1d or curr_close < curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA34 (trend change) OR close above Donchian middle (mean reversion)
            if curr_close > curr_ema_1d or curr_close > curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close above Donchian high AND price above 1d EMA34 AND volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Close below Donchian low AND price below 1d EMA34 AND volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals